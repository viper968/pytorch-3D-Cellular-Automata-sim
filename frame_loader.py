import bpy
import json
import numpy as np

# =========================================================================
# GENERATIONS FORMAT LOADER (v4 - metadata & population aware)
# =========================================================================
# Changes vs v3 (per handoff v2):
#   - Format detection now requires the embedded 'metadata' JSON entry and
#     checks its format tag ("generations-delta-v1"). Archives with old
#     born/died keys OR no metadata are rejected with a clear error.
#   - MAX_POOL_SIZE comes straight from visible_population.max() — the
#     incremental peak-count logic is deleted entirely.
#   - MAX_STATE comes from meta["num_states"] - 1 (exact), with the old
#     scan-derived value kept only as a consistency check fallback.
#   - Grid extents come from meta["grid_size"][::-1] (stored coords are
#     column-flipped by the simulator, so the reversed list gives exact
#     per-column extents). The max-coord scan is gone — it could undershoot
#     when edge cells never light up; metadata can't.
#   - Timeline length comes from meta["total_frames"] instead of parsing
#     f_{n} key names.
#   - Rule string, seed, termination reason, and a population summary are
#     printed on import for render traceability.
#   - The pre-pass now exists ONLY to build scrubbing keyframes (pure delta
#     replay + periodic snapshots) — no counting work per frame.
# =========================================================================

# =========================================================================
# CONFIGURATION
# =========================================================================
NPZ_PATH = "/home/simon/scripts/cell_auto/ca_batch_exports/run_011/data.npz"
UPDATE_EVERY_X_FRAMES = 2

# How often (in simulation steps) a full (indices, states) snapshot is kept
# in RAM. Lower = faster far seeks, more memory. Higher = the opposite.
KEYFRAME_INTERVAL = 50

# ---- Automatic camera rig positioning ----
# Ratios derived from the known-good reference framing on a 100^3 grid:
# target at (50, 50, 38) with the camera 260 units along -Y from it.
AUTO_POSITION_CAMERA = True
CAMERA_TARGET_NAME = "camera_target"     # empty/object the camera orbits around
CAMERA_NAME = "Camera"                    # falls back to the active scene camera
TARGET_RATIO = (0.50, 0.50, 0.38)         # target pos as a fraction of grid extents
CAMERA_DISTANCE_FACTOR = 2.6              # distance = factor * max grid extent

SIM_DATA = np.load(NPZ_PATH)

# =========================================================================
# FORMAT DETECTION (metadata-first, per handoff v2 §3.5)
# =========================================================================
_old_key = next((k for k in SIM_DATA.files if "_born_" in k or "_died_" in k), None)
if _old_key is not None:
    raise RuntimeError(
        f"'{NPZ_PATH}' is an OLD-format archive (found key '{_old_key}'). "
        "This loader only reads the Generations delta+states format. "
        "Re-export the run with the updated simulator."
    )
if "metadata" not in SIM_DATA.files:
    raise RuntimeError(
        f"'{NPZ_PATH}' has no 'metadata' entry — this is an old-format archive. "
        "Re-export the run with the updated simulator."
    )

META = json.loads(str(SIM_DATA["metadata"]))

if META.get("format") != "generations-delta-v1":
    raise RuntimeError(
        f"Unsupported format tag '{META.get('format')}' in '{NPZ_PATH}'. "
        "This loader expects 'generations-delta-v1'."
    )

# =========================================================================
# EXACT PARAMETERS FROM METADATA (replaces all auto-detection scans)
# =========================================================================
# Stored coord rows are [d2_idx, d1_idx, d0_idx] relative to
# grid_size = [d0, d1, d2] (the simulator flips columns before saving),
# so reversing grid_size gives exact per-column extents.
_col_dims = META["grid_size"][::-1]
GRID_SIZE_X = int(_col_dims[0])   # extent of stored column 0
GRID_SIZE_Y = int(_col_dims[1])   # extent of stored column 1
GRID_SIZE_Z = int(_col_dims[2])   # extent of stored column 2

stride_x = GRID_SIZE_Y * GRID_SIZE_Z
stride_y = GRID_SIZE_Z
TOTAL_GRID_CELLS = GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z

TOTAL_FRAMES = int(META["total_frames"])          # sim steps f_0 .. f_{TOTAL_FRAMES-1}
max_step_idx = TOTAL_FRAMES - 1
MAX_STATE = int(META["num_states"]) - 1           # exact, no derivation needed

# MAX_POOL_SIZE is free now: visible_population[i] is exactly the visible
# cell count at frame i, so its max is the peak pool requirement.
VISIBLE_POPULATION = SIM_DATA["visible_population"]
POPULATION = SIM_DATA["population"]
MAX_POOL_SIZE = int(VISIBLE_POPULATION.max())

final_blender_frame = max_step_idx * UPDATE_EVERY_X_FRAMES

# =========================================================================
# IMPORT SUMMARY (render traceability)
# =========================================================================
print("=" * 60)
print(f"[*] Format:        {META['format']}")
print(f"[*] Rule:          {META['rule_string']}  "
      f"(neighborhood: {META['neighborhood_type']}, r={META['neighborhood_radius']}, "
      f"boundary: {META['boundary_mode']})")
print(f"[*] Seed:          mode={META['seed_mode']}, prob_alive={META['prob_alive']}, "
      f"run_seed={META['run_seed']}, run_index={META['run_index']}")
print(f"[*] Grid (sim):    {META['grid_size']}  ->  stored-column extents: "
      f"{GRID_SIZE_X}x{GRID_SIZE_Y}x{GRID_SIZE_Z}")
print(f"[*] Frames:        {TOTAL_FRAMES}  |  Terminated: {META['termination_reason']}")
print(f"[*] Num States:    {META['num_states']}  (MAX_STATE = {MAX_STATE} -> "
      f"use as 'From Max' in the GN Map Range node)")
print(f"[*] Population:    occupied peak {int(POPULATION.max())}, "
      f"final {int(POPULATION[-1])}  |  visible peak {MAX_POOL_SIZE}, "
      f"final {int(VISIBLE_POPULATION[-1])}")
print("=" * 60)

if bpy.context.scene:
    bpy.context.scene.frame_end = final_blender_frame
else:
    for scene in bpy.data.scenes:
        scene.frame_end = final_blender_frame
print(f"[*] Timeline End set to Frame: {final_blender_frame}")

# =========================================================================
# FRAME ARRAY LOADING (coords + row-aligned states, mixed precision)
# =========================================================================


def load_frame_delta(step_idx):
    """
    Returns (flat_indices int32, new_states uint8), row-aligned, combining
    the 8-bit and 16-bit precision tiers. For step 0 the coords come from
    the 'initial' keys; for later steps from the 'delta' keys.
    """
    if step_idx == 0:
        coord_keys = ("f_0_initial_8", "f_0_initial_16")
    else:
        coord_keys = (f"f_{step_idx}_delta_8", f"f_{step_idx}_delta_16")
    state_keys = (f"f_{step_idx}_states_8", f"f_{step_idx}_states_16")

    idx_parts, state_parts = [], []
    for ck, sk in zip(coord_keys, state_keys):
        if ck in SIM_DATA:
            arr = SIM_DATA[ck].astype(np.int32)
            if arr.size > 0:
                if sk not in SIM_DATA or SIM_DATA[sk].shape[0] != arr.shape[0]:
                    raise RuntimeError(
                        f"Archive corrupt: '{ck}' has {arr.shape[0]} coords but "
                        f"'{sk}' is missing or misaligned."
                    )
                idx_parts.append(arr[:, 0] * stride_x + arr[:, 1] * stride_y + arr[:, 2])
                state_parts.append(SIM_DATA[sk].astype(np.uint8))

    if not idx_parts:
        return (np.array([], dtype=np.int32), np.array([], dtype=np.uint8))
    if len(idx_parts) == 1:
        return idx_parts[0], state_parts[0]
    return np.concatenate(idx_parts), np.concatenate(state_parts)


# =========================================================================
# PRE-PASS: keyframe building only (counting now comes from metadata)
# =========================================================================
print(f"[*] Building scrub keyframes every {KEYFRAME_INTERVAL} steps...")

STATE_GRID = np.zeros(TOTAL_GRID_CELLS, dtype=np.uint8)  # flat 1D dense grid
KEYFRAME_CACHE = {}  # step -> (indices int32, states uint8)

for step in range(max_step_idx + 1):
    idx, states = load_frame_delta(step)
    if idx.size > 0:
        STATE_GRID[idx] = states

    if step % KEYFRAME_INTERVAL == 0:
        kf_indices = np.nonzero(STATE_GRID)[0].astype(np.int32)
        KEYFRAME_CACHE[step] = (kf_indices, STATE_GRID[kf_indices].copy())

# Sanity check: reconstructed final visible count must match the stat array
_final_recon = int((STATE_GRID != 0).sum())
_final_meta = int(VISIBLE_POPULATION[max_step_idx])
if _final_recon != _final_meta:
    print(f"[!] WARNING: reconstructed final visible count ({_final_recon}) "
          f"!= visible_population[-1] ({_final_meta}). Archive may be corrupt.")

# Rewind the dense grid to step 0 so playback starts clean
STATE_GRID[:] = 0
_kf0_idx, _kf0_states = KEYFRAME_CACHE[0]
STATE_GRID[_kf0_idx] = _kf0_states
STATE_GRID_STEP = 0

print(f"[*] Cached {len(KEYFRAME_CACHE)} keyframes. "
      f"Pool size {MAX_POOL_SIZE} (from visible_population).")

# =========================================================================
# CAMERA RIG AUTO-POSITIONING
# =========================================================================


def setup_camera_rig():
    """
    Places the orbit target inside the grid and sets the camera's distance
    from it, scaled from the reference framing (100^3 grid: target at
    (50, 50, 38), camera 260 units along -Y).

    Handles both rig styles:
      - Camera PARENTED to the target (orbit-by-rotating-target): the camera
        gets a LOCAL offset of (0, -distance, 0), so rotating the target
        still sweeps it around the grid.
      - Unparented camera: it's placed in world space at target + (0,
        -distance, 0), leaving its rotation untouched.
    """
    target = bpy.data.objects.get(CAMERA_TARGET_NAME)
    if target is None:
        print(f"[!] Camera helper: no object named '{CAMERA_TARGET_NAME}' found — "
              f"skipping. (Set CAMERA_TARGET_NAME to match your rig.)")
        return

    cam = bpy.data.objects.get(CAMERA_NAME)
    if cam is None and bpy.context.scene:
        cam = bpy.context.scene.camera
    if cam is None:
        print(f"[!] Camera helper: no camera named '{CAMERA_NAME}' and no active "
              f"scene camera — skipping.")
        return

    target_pos = (
        GRID_SIZE_X * TARGET_RATIO[0],
        GRID_SIZE_Y * TARGET_RATIO[1],
        GRID_SIZE_Z * TARGET_RATIO[2],
    )
    distance = CAMERA_DISTANCE_FACTOR * max(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z)

    target.location = target_pos

    if cam.parent == target:
        cam.location = (0.0, -distance, 0.0)
        rig_style = "parented (local -Y offset)"
    else:
        cam.location = (target_pos[0], target_pos[1] - distance, target_pos[2])
        rig_style = "unparented (world position)"

    print(f"[*] Camera rig: target '{target.name}' -> "
          f"({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}), "
          f"camera '{cam.name}' at distance {distance:.0f} [{rig_style}]")


if AUTO_POSITION_CAMERA:
    setup_camera_rig()

# =========================================================================
# RECONSTRUCTION: dense state grid + keyframe seeks
# =========================================================================
LAST_COMPUTED_STEP = -1
LAST_COMPUTED_RESULT = None  # (indices int32, states uint8)


def get_state_for_sim_step(step_idx):
    """
    Returns (flat_indices int32, states uint8) for the visible cells at a
    zero-indexed simulation step. Sequential forward playback advances the
    persistent dense grid one delta at a time; far seeks restore from the
    nearest earlier keyframe and replay at most KEYFRAME_INTERVAL deltas.
    """
    global STATE_GRID_STEP, LAST_COMPUTED_STEP, LAST_COMPUTED_RESULT

    step_idx = max(0, min(step_idx, max_step_idx))

    if step_idx == LAST_COMPUTED_STEP:
        return LAST_COMPUTED_RESULT

    nearest_kf = (step_idx // KEYFRAME_INTERVAL) * KEYFRAME_INTERVAL
    if nearest_kf not in KEYFRAME_CACHE:
        available = [k for k in KEYFRAME_CACHE if k <= step_idx]
        nearest_kf = max(available) if available else 0

    # Continue from the grid's current position when that's the shorter path
    # (the common case: normal forward playback, one step at a time).
    if 0 <= STATE_GRID_STEP <= step_idx and (step_idx - STATE_GRID_STEP) <= (step_idx - nearest_kf):
        start_step = STATE_GRID_STEP
    else:
        STATE_GRID[:] = 0
        kf_idx, kf_states = KEYFRAME_CACHE[nearest_kf]
        STATE_GRID[kf_idx] = kf_states
        start_step = nearest_kf

    for s in range(start_step + 1, step_idx + 1):
        idx, states = load_frame_delta(s)
        if idx.size > 0:
            STATE_GRID[idx] = states

    STATE_GRID_STEP = step_idx

    active_indices = np.nonzero(STATE_GRID)[0].astype(np.int32)
    active_states = STATE_GRID[active_indices]

    LAST_COMPUTED_STEP = step_idx
    LAST_COMPUTED_RESULT = (active_indices, active_states)
    return LAST_COMPUTED_RESULT


# =========================================================================
# MESH UPDATE HANDLER
# =========================================================================
LAST_SIM_STEP = -1


def update_mesh_handler(scene):
    global LAST_SIM_STEP
    frame = scene.frame_current
    sim_step = frame // UPDATE_EVERY_X_FRAMES

    if sim_step == LAST_SIM_STEP:
        return

    obj = bpy.data.objects.get("CA_Point_Cloud")
    if not obj:
        return

    indices, active_states = get_state_for_sim_step(sim_step)
    num_active = len(indices)

    # 1. Static pool array pre-filled with the hidden dummy position
    full_coords = np.full((MAX_POOL_SIZE, 3), [0.0, 0.0, -10000.0], dtype=np.float32)

    # 2. Overwrite the top chunk with real active coordinates via stride unflattening
    if num_active > 0:
        x = indices // stride_x
        y = (indices % stride_x) // stride_y
        z = indices % stride_y
        full_coords[:num_active] = np.column_stack((x, y, z))

    mesh = obj.data

    # 3. Structural allocation happens ONCE, keeping Cycles stable
    if len(mesh.vertices) != MAX_POOL_SIZE:
        mesh.clear_geometry()
        mesh.vertices.add(MAX_POOL_SIZE)

    # 3b. Per-vertex state attribute for Geometry Nodes shading.
    #     (Re-checked every time since clear_geometry wipes attributes.)
    if "state" not in mesh.attributes:
        mesh.attributes.new(name="state", type='FLOAT', domain='POINT')

    # 4. Stream coordinates + states to the mesh via C-level foreach_set
    mesh.vertices.foreach_set("co", full_coords.flatten())

    state_values = np.zeros(MAX_POOL_SIZE, dtype=np.float32)
    if num_active > 0:
        state_values[:num_active] = active_states.astype(np.float32)  # raw 1..MAX_STATE
    mesh.attributes["state"].data.foreach_set("value", state_values)

    mesh.update()
    LAST_SIM_STEP = sim_step


# Register the handler
bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(update_mesh_handler)