Here is the comprehensive technical breakdown and documentation for your Blender `frame_loader.py` script (v4 — Generations format, metadata-aware).

---

### **Overview: Blender CA Point Cloud Importer**

This script is the bridge between the GPU-accelerated Generations-style Cellular Automata engine and Blender. It reads the delta-compressed `.npz` archives (multi-state format, `generations-delta-v1`), reconstructs any frame on demand using a dense state grid plus keyframe snapshots, and injects coordinates AND per-cell state values into a Blender mesh using the "Persistent Pool" technique — guaranteeing stable Cycles performance and enabling state-driven shading through Geometry Nodes.

---

### **1. Configuration Constants (top of file)**

* **`NPZ_PATH`** — path to the exported `data.npz` archive.
* **`UPDATE_EVERY_X_FRAMES`** — timing multiplier; sim step n displays on Blender frame n × this value.
* **`KEYFRAME_INTERVAL`** — how often (in sim steps) a full `(indices, states)` snapshot is cached in RAM. Lower = faster timeline scrubbing, more memory. Higher = the opposite. Default 50.
* **`AUTO_POSITION_CAMERA`** / **`CAMERA_TARGET_NAME`** / **`CAMERA_NAME`** / **`TARGET_RATIO`** / **`CAMERA_DISTANCE_FACTOR`** — camera rig auto-positioning (section 5).

---

### **2. Format Detection & Metadata**

The loader ONLY accepts the new Generations format and refuses anything else loudly:

* Any archive containing `_born_` / `_died_` keys → RuntimeError (old format, re-export).
* Any archive missing the `metadata` entry → RuntimeError (old format, re-export).
* `metadata` present but `format != "generations-delta-v1"` → RuntimeError (unsupported version).

The embedded metadata JSON (a 0-d numpy unicode array, no pickle needed) replaces every auto-detection scan the old loader performed:

* **Grid extents** — `meta["grid_size"][::-1]`. The simulator flips coordinate columns before saving (`torch.nonzero().flip(dims=[1])`), so stored column 0 spans `grid_size[2]`, column 1 spans `grid_size[1]`, column 2 spans `grid_size[0]`. Reversing the list gives exact per-column extents. Unlike the old max-coordinate scan, this can never undershoot when edge cells stay dark.
* **`MAX_STATE`** — `meta["num_states"] - 1`. Fully alive / freshly born cells hold this value; 1..MAX_STATE-1 is the refractory countdown; 0 = dead/hidden.
* **Timeline length** — `meta["total_frames"]` (steps f_0 .. f_{total_frames-1}); no more parsing key names.
* **`MAX_POOL_SIZE`** — `visible_population.max()`. The stat array gives the exact visible cell count per frame, so the peak pool requirement is a single array read. The entire reconstruction-based peak-count pre-pass is gone.

On import the loader prints a traceability summary: format tag, rule string, neighborhood type/radius, boundary mode, seed mode/probability/seed/run index, grid size, frame count, termination reason, num_states (with the MAX_STATE value to type into the GN Map Range node), and occupied-vs-visible population peaks/finals.

---

### **3. Core Functions & Optimizations**

**`load_frame_delta(step_idx)`**

* **What it does:** Fetches one frame's coordinate + state data. Step 0 reads the `f_0_initial_8/16` full snapshot; later steps read `f_{n}_delta_8/16`. Each coords array has a row-aligned `f_{n}_states_8/16` companion carrying the NEW state value for that cell.
* **Mixed-Precision Combiner:** transparently merges the 8-bit and 16-bit precision tiers (coords with all axes < 256 vs. any axis ≥ 256), casting up to int32 and flattening `[c0,c1,c2]` triplets into 1D indices via row-major strides.
* **Integrity check:** raises a clear "archive corrupt" error if a states array is missing or misaligned with its coords array.

**Dense-grid reconstruction (`STATE_GRID` + `get_state_for_sim_step`)**

* **What it does:** Maintains one persistent flat uint8 array covering the whole grid. Applying a frame is a single fancy-indexed assignment — `STATE_GRID[idx] = states` — which handles births (0 → MAX_STATE), deaths (→ 0), and refractory decay ticks (e.g. 3 → 2) in one operation. Visible cells at any moment are `np.nonzero(STATE_GRID)`, and their states are read off the same grid, guaranteed index-aligned.
* **Why it's fast:** cost per frame is O(size of the delta), not O(size of the active set) — no sorting, no set math, no `setdiff1d`.

**Keyframe cache (`KEYFRAME_CACHE`)**

* **What it does:** Every `KEYFRAME_INTERVAL` steps, a sparse `(indices, states)` snapshot is stored. Snapshots must include states — indices alone cannot restore refractory levels.
* **Seeking:** normal forward playback advances the persistent grid one delta at a time (near-zero cost). Backward scrubs or long jumps restore the nearest earlier keyframe and replay at most `KEYFRAME_INTERVAL` deltas. Memory stays bounded regardless of run length.
* **When it's built:** eagerly, in a single pre-pass at load time (pure delta replay — no counting work, since all counts now come from metadata). The pre-pass ends with a free sanity check: the reconstructed final visible count is compared against `visible_population[-1]`, printing a corruption warning on mismatch. The grid is then rewound to step 0 for playback.

**`update_mesh_handler(scene)`**

* **What it does:** The `frame_change_pre` callback. Converts the Blender frame to a sim step, fetches `(indices, states)`, un-flattens indices to XYZ via stride arithmetic, and streams everything to the mesh.
* **The Persistent Pool:** a fixed `MAX_POOL_SIZE`-vertex allocation; active cells occupy the top of the pool, dead vertices park at Z = -10000. This bypasses the Cycles BVH rebuild choke — the BVH is merely refit each frame.
* **State attribute:** alongside `foreach_set("co", ...)`, the handler writes a per-vertex FLOAT point attribute named `"state"` via `foreach_set("value", ...)`, carrying raw values 1..MAX_STATE for active cells and 0 for pooled dead vertices. The attribute existence check runs every frame because `clear_geometry()` wipes attributes.
* **C-Level Data Transfer:** both writes are low-level `foreach_set` calls — raw NumPy streams, no Python loops.

---

### **4. Data Flow**

1. **Load & validate:** open the `.npz`, reject old formats, parse metadata, derive all parameters (grid extents, strides, MAX_STATE, MAX_POOL_SIZE, timeline length), print the import summary, set `scene.frame_end`.
2. **Keyframe pre-pass:** replay every delta once through the dense grid, snapshotting `(indices, states)` every `KEYFRAME_INTERVAL` steps; verify the final count against `visible_population[-1]`; rewind to step 0.
3. **Camera rig:** if enabled, position the orbit target and camera (section 5).
4. **Playback:** each Blender frame change triggers the handler → `get_state_for_sim_step` → dense-grid advance or keyframe seek → unflatten → pool write + state attribute write → `mesh.update()`.

---

### **5. Camera Rig Auto-Positioning**

Scales a known-good reference framing (100³ grid: target at (50, 50, 38), camera 260 units along -Y) to any grid size:

* Target position = `(0.50·X, 0.50·Y, 0.38·Z)` of the stored-column grid extents (`TARGET_RATIO`).
* Camera distance = `2.6 × max(X, Y, Z)` (`CAMERA_DISTANCE_FACTOR`).

Handles both rig styles: if the camera is parented to the target (orbit-by-rotating-target), it receives a LOCAL offset `(0, -distance, 0)` so the orbit still works; otherwise it is placed in world space at `target + (0, -distance, 0)` with rotation untouched. Missing objects produce a skip message naming what was searched for, never an error. Disable entirely with `AUTO_POSITION_CAMERA = False`.

---

### **6. Inputs and Outputs**

**Inputs:**

* A `generations-delta-v1` `.npz` archive containing: `f_0_initial_8/16` + `f_0_states_8/16`, per-frame `f_{n}_delta_8/16` + `f_{n}_states_8/16`, the `metadata` JSON entry, and the `population` / `visible_population` stat arrays.
* A scene object named exactly `"CA_Point_Cloud"` to receive the geometry.
* Optionally, a camera-orbit empty (default name `"camera_target"`) and camera (default `"Camera"`).

**Outputs:**

* Timeline scaled to `(total_frames - 1) × UPDATE_EVERY_X_FRAMES`.
* A persistent-pool point cloud with per-vertex `"state"` FLOAT attribute, ready for Geometry Nodes: filter points below Z = -9999, instance cubes, then Named Attribute("state") → Map Range (From 1..MAX_STATE, To 0..1) → Color Ramp → Store Named Attribute (Color, Instance domain, e.g. "col") → read in the material with an Attribute node set to type "Instancer". Realize Instances is NOT needed. Optionally drive instance Scale from the same Map Range so decaying cells shrink.
* Console import summary tying every render back to the exact rule string and seed that produced it.
