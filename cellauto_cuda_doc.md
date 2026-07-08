---

### **Overview: PyTorch 3D Cellular Automata Engine (Generations Edition)**

This script is a high-performance, GPU-accelerated batch simulator for 3D Cellular Automata. It
computes physics entirely on the GPU with PyTorch, supports the standard "Generations" rule
family (multi-state cells with refractory decay) in configurable Moore / von Neumann
neighborhoods, culls hidden geometry with a three-culler system, compresses per-frame deltas
in parallel across CPU threads, and streams self-documenting `.npz` archives to disk through
background I/O workers. Each run also produces a `run_overview.png` dashboard graph.

---

### **1. Simulation Rules & Configuration**

#### Rule definition (three equivalent ways)
* **Manual variables:** `SURVIVE_COUNTS`, `BIRTH_COUNTS`, `NUM_STATES`, `NEIGHBORHOOD_TYPE`,
  `NEIGHBORHOOD_RADIUS`.
* **`RULE_STRING`** — a Visions-of-Chaos style `S/B/C/N` string (e.g. `"4/4/5/M"`,
  `"9-26/5-7,12-13,15/5/M"`, `"/4/2/M"`). Supports comma lists, hyphen ranges, empty survival
  sections, and `M` / `VN` neighborhood tokens. Overrides the manual variables when set.
* **`RULE_PRESETS`** — named rules from Softology's 3D CA blog: `445`, `amoeba`,
  `architecture`, `clouds-1`, `3d-brain`, `pyroclastic`, and `original` (this script's
  historical hardcoded rule, `6-8/5/2/M`). `RULE_STRING` accepts a preset name directly.

#### Generations-style states (`NUM_STATES`)
`0` = dead, `NUM_STATES-1` = fully alive. Values between are a forced refractory countdown:
a decaying cell drops exactly 1 state per frame regardless of neighbors and cannot be reborn
until it reaches 0. **Only fully-alive cells count as neighbors** for birth/survival checks
(standard Generations / Brian's Brain convention). `NUM_STATES = 2` reproduces classic
boolean behavior exactly (verified bit-identical).

#### Neighborhoods (`build_neighborhood_kernel`)
Generates the convolution kernel from config: `"moore"` = full cube within Chebyshev distance
≤ radius (26 cells at r=1), `"von_neumann"` = Manhattan distance ≤ radius (6 cells at r=1).
Arbitrary radii supported (Moore r=2 = 124 neighbors).

#### Seeding (`SEED_MODE`, `build_initial_alive_grid`)
* `"soup"` — whole grid random at `PROB_ALIVE` (original behavior)
* `"center_block"` — centered `SEED_BLOCK_SIZE`³ random region (required in practice for
  growth rules like 445; full soup makes them die or explode)
* `"single"` — one cell at grid center
* `"shell"` — hollow 1-cell-thick centered box at `PROB_ALIVE` density
Oversized blocks clamp to the grid instead of erroring.

#### Boundary (`BOUNDARY_MODE`)
`"wrap"` = torus (circular padding, original behavior); `"dead"` = bounded walls (constant-0
padding) — expanding rules flatten at walls instead of self-colliding.

#### Rule search / mining (`RULE_SEARCH_MODE`)
Randomizes survive/birth/states per run (birth never contains 0, which would instantly flood
a wrapped grid). The rule is printed per run, embedded in metadata, and listed in the batch
summary, so long-lived discoveries are exactly reproducible from `run_seed` + rule string.
Fully deterministic under `RANDOM_SEED` (the `random` module is seeded per run alongside
numpy/torch).

---

### **2. Physics Core**

#### `calculate_next_gen_tensor(state_tensor)`
The grid is a `uint8` **state tensor** (not boolean). Per step, entirely on GPU:
1. `alive_mask = (state == MAX_STATE)` — only fully-alive cells are counted.
2. Neighbor counts via one `F.conv3d` over the configured kernel (bf16, padded per
   `BOUNDARY_MODE`), rounded and cast to int16 so float noise can never shift a count.
3. Dead cells matching `BIRTH_COUNTS` → `MAX_STATE`; alive cells failing `SURVIVE_COUNTS` →
   `MAX_STATE-1`; refractory cells → state−1 (clamped, so uint8 can never wrap even though
   `torch.where` evaluates both branches everywhere).

---

### **3. The Three-Culler System**

Culling operates on **occupancy** (any non-zero state) and decides which cells are worth
sending to Blender each frame.

#### Phase 1 — `get_visible_hull_mask` (FLOOD depth peel, up to 18 ray directions)
Keeps cells within `SHELL_THICKNESS` occupied cells of the grid boundary along each viewing
ray, shrinking thickness until the result fits `MAX_PHASE1_POINTS`. Ray directions:
* **6 face rays** (always on) — the original axis cumsums, hoisted out of the thickness-shrink
  loop and computed once per frame.
* **`FLOOD_DIAGONALS`** — +12 edge rays at 45° (e.g. `(1,1,0)`). Implemented by *shearing* the
  grid (pad+reshape reflow — no gather ops) so diagonals become straight columns, then reusing
  plain `cumsum`. Catches staircase/angled geometry the axis rays wrongly cull (+26% genuinely
  visible cells at t=1 in testing). ~190 MB extra VRAM at 400×50×400. Measured cost on the
  RTX 3050 Ti @ 200³: ~22-25 FPS faces-only vs ~8 FPS with edges (phase-1 frames only).
All 18 directions were validated cell-for-cell against brute-force ray marching. (Corner rays
were prototyped and removed: 8 more directions via double shears cost ~4x more FPS than the
edges while catching fewer extra cells — the worst accuracy-per-cost by far.) The script sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch import — the culler's
variable-size intermediates otherwise fragment the caching allocator into spurious OOMs.

#### Phase 2 option A — `get_basic_surface_mask` ("basic")
Keeps any occupied cell touching an empty cell. Single fast pass, but also keeps the walls of
fully sealed internal cavities that can never be seen.

#### Phase 2 option B — `get_exterior_hull_mask` ("exterior", default)
The exact true hollow shape: flood-fills air from the grid boundary via 6-connected geodesic
dilation (`_dilate6`, boolean slice-ORs), then keeps only occupied cells touching
**outside-connected** air. Sealed cavities are culled; open pockets and bent tunnels are
correctly kept (the basic and FLOOD cullers each fail one of those). Strictly ≤ basic's cell
count → smaller files. Cost control: full cold solve on the first phase-2 frame and every
`EXTERIOR_RECONVERGE_EVERY` frames; all other frames warm-start from the previous frame's
exterior air and run only `EXTERIOR_WARM_ITERS` dilation steps. Warm-start error is
transient, tiny, and provably only ever *over-shows* (never punches holes). Selected via
`PHASE2_CULLER`. Warm state resets per run.

#### The validated transition
Crossing `CAPACITY_THRESHOLD` no longer hard-switches cullers (which caused a visible jump in
remaining-cell counts). Instead the FLOOD baseline count is frozen at the switch moment, and
each frame the phase-2 culler must match or beat that baseline before being trusted; until
then the frame is redone with FLOOD (`FLOOD (fallback)` in the console). Once caught up, the
phase-2 culler is locked in permanently.

---

### **4. Delta Format & Output Archive (`generations-delta-v1`)**

Frame 0 stores a full snapshot; every later frame stores only cells whose **visible state**
changed (captures births, deaths, and refractory ticks in one entry type — the old
`born`/`died` arrays no longer exist):

* `f_0_initial_8/16` (N,3 coords) + `f_0_states_8/16` (N, row-aligned uint8 states)
* `f_{n}_delta_8/16` (changed coords) + `f_{n}_states_8/16` (NEW state; 0 = remove/hide)
* `metadata` — a JSON blob (0-d unicode array, no pickle needed) with the full run config:
  rule string, S/B/C, neighborhood, boundary, grid size, seed settings, run seed, culler
  settings, total frames, termination reason. Recover with
  `json.loads(str(np.load(path)["metadata"]))`.
* `population` / `visible_population` — int64 per-frame cell counts (pre- and post-culling).
  `visible_population.max()` gives the loader its vertex pool size for free.

Mixed precision is preserved: coords with all axes < 256 go to `_8` arrays, others to `_16`,
with the state arrays split by the same row mask so rows stay aligned within each tier.
Reconstruction on the loader side is one fancy-indexed assignment per frame (verified with a
full round-trip test). **Coordinate columns are flipped relative to `grid_size`** (stored
column 0 spans `grid_size[2]`) — see the frame_loader handoff doc.

The critical `last_mask` delta bug is fixed: deltas are now computed against the true previous
frame (they were frozen against frame 0, which both bloated files ~10x and produced permanent
ghost cells on reconstruction).

---

### **5. I/O Pipeline & Memory Management**

#### Parallel compression (`_compress_entry` + `_write_zip_manual`)
Each run's arrays are deflate-compressed **in parallel** across
`COMPRESSION_THREADS_PER_FILE` (=16) threads — `zlib` releases the GIL during compression, so
this is genuine multi-core speedup. The compressed entries are then assembled sequentially by
a hand-rolled ZIP writer (local headers + central directory + EOCD), which exists because
`zipfile.ZipFile` cannot accept pre-compressed bytes. Output is byte-identical to the old
sequential method and fully `np.load`-compatible.

#### Background workers
`NUM_WORKERS` (=4) threads drain a bounded queue (`MAX_CACHE_SIZE` = 2, deliberately small so
the GPU loop throttles instead of stacking whole runs in RAM).

#### RAM release (`_release_freed_memory_to_os`)
glibc's malloc adaptively raises its mmap threshold under repeated large frees, causing freed
numpy/zlib buffers to be retained in the process heap indefinitely (RAM climbed ~5 GB per run
before the fix). After each archive is written, the worker explicitly deletes every reference
(`task, run_dict, entries, futures` — Future objects secretly retain result copies) and calls
`malloc_trim(0)` via ctypes, returning the memory to the OS immediately.

---

### **6. Instrumentation**

* **Console status line:** frame index, **FPS** (rolling average over `FPS_AVG_WINDOW`
  frames), live cells, active culler (`FLOOD` / `FLOOD (fallback)` / `EXTERIOR` / `BASIC` /
  `... (caught up)`), culler stats (thickness, remaining, transition baseline), zip queue
  depth.
* **`run_overview.png`** (per run, toggle `GENERATE_RUN_GRAPH`; needs matplotlib, skips
  gracefully without it): three shared-axis panels — populations with the oscillator
  threshold, threshold-hit and phase-2-graduation markers, and shaded transition window; FPS
  with mean; per-frame delta entry counts with FLOOD thickness steps on a twin axis. Title
  and summary carry the full config: rule, grid, states, neighborhood, boundary, seed setup,
  run seed, start/peak counts, wall time, culler toggles, and termination reason. Plot
  failures warn but can never kill a batch.
* **Batch summary:** per-run frame count, rule string, and outcome.

---

### **7. Termination & Determinism**

Runs end on: `MAX_FRAMES` reached, total extinction, or oscillator detection (full state
tensor hashed **every frame** — rules like amoeba can stabilize at a dense, above-threshold
grid, and the fixed-size transfer costs the same at any density; a repeat count of
`REPEAT_CUTOFF` triggers "Stabilized"). Hashing the *integer state* tensor (not just
occupancy) is required for multi-state rules — same occupancy with different refractory
levels is a different state. The capacity threshold is likewise **not a latch**: density is
re-evaluated every frame, so growth rules that re-densify revert to FLOOD (restoring the
point cap) and revalidate against a fresh baseline on the next descent; runs that start
below the threshold skip validation (no FLOOD handoff = no jump to hide).
`RANDOM_SEED` seeds `random`, numpy, and torch per run (`RANDOM_SEED + run_idx`), making
every run — including rule-search draws and seeding — exactly reproducible.

---

### **8. Inputs and Outputs**

Configured entirely via the constants block at the top of the file (sections 1–1h): grid,
rule, seeding, boundary, cullers, search mode, I/O, instrumentation.

Output tree per batch: `BASE_OUTPUT_DIR/run_NNN/` containing **`data.npz`** (the delta
archive described in §4) and **`run_overview.png`**. Extinction runs are optionally deleted
(`DELETE_EXTINCTION_RUNS`).
