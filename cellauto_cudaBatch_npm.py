#!/home/simon/scripts/cell_auto/cell_auto/bin/python
import numpy as np
import os
# Must be set before torch initializes CUDA: expandable segments prevents the
# allocator fragmentation OOMs caused by the culling pipeline's variable-size
# intermediate tensors ("reserved but unallocated" memory ballooning).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import json
import random
import queue
import threading
import shutil
import torch
import torch.nn.functional as F
import io
import zipfile
import zlib
import struct
import ctypes
from concurrent.futures import ThreadPoolExecutor

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None

def _release_freed_memory_to_os():
    """glibc's malloc adaptively raises its mmap threshold after seeing
    repeated large frees, which causes big numpy/zlib buffers to start
    living in the process heap instead of being mmap'd -- and heap memory
    can only be returned to the OS from the END of the heap, so freed
    blocks in the middle just sit there forever even though Python has
    already dropped every reference to them. malloc_trim(0) explicitly
    asks glibc to scan for and release any freed memory it's holding."""
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass

# ==========================================
# 1. Global Batch & Simulation Configuration
# ==========================================
TOTAL_SIMULATION_RUNS = 15
DELETE_EXTINCTION_RUNS = False

GRID_SIZE = (200, 200, 200)
if isinstance(GRID_SIZE, int):
    GRID_SIZE = (GRID_SIZE, GRID_SIZE, GRID_SIZE)

PROB_ALIVE = 0.15
MAX_FRAMES = 0
BASE_OUTPUT_DIR = "/home/simon/scripts/cell_auto/ca_batch_exports"

CAPACITY_THRESHOLD = False
SHELL_THICKNESS = 3
MAX_PHASE1_POINTS = 1141360

# Phase-1 FLOOD culler ray directions:
#   False -> 6 cardinal rays (faces only): the original axis-aligned depth peel.
#   True  -> 18 rays (6 faces + 12 edge diagonals at 45 degrees): catches cells
#            visible from diagonal viewing angles that pure axis rays cull
#            (staircase surfaces, angled pockets). Keeps more cells per
#            thickness (a closer match to the true visible hull), so the
#            MAX_PHASE1_POINTS thickness-shrink may engage a bit more often.
#            Costs 12 extra cumsum passes (+ cheap shear reshapes) per frame.
#            Measured on RTX 3050 Ti @ 200^3: ~22-25 FPS faces-only vs ~8 FPS
#            with edges on (phase-1 frames only), and ~190MB extra VRAM at
#            400x50x400 for the hoisted count tensors.
FLOOD_DIAGONALS = True

REPEAT_CUTOFF = 2
RANDOM_SEED = 42

# Console FPS metric: frames/s averaged over this many recent frames.
FPS_AVG_WINDOW = 50
# After each run, save a "run_overview.png" graph into the run folder next to
# data.npz: populations, culler phases, FPS, per-frame delta sizes, and the
# full run configuration. Requires matplotlib (pip install matplotlib);
# gracefully skipped with a warning if it isn't installed.
GENERATE_RUN_GRAPH = True

# ==========================================
# 1b. CA Rule Configuration (Neighborhood + "Generations"-style states)
# ==========================================
# Neighborhood shape used to count neighbors for the birth/survival rule.
# "moore"       -> full cube of cells around the center (Chebyshev distance <= radius).
#                  radius=1 is the classic 26-cell 3D Moore neighborhood.
# "von_neumann" -> only face-connected cells (Manhattan distance <= radius).
#                  radius=1 is the classic 6-cell 3D von Neumann neighborhood.
NEIGHBORHOOD_TYPE = "moore"
NEIGHBORHOOD_RADIUS = 1

# Standard Life-like B/S rule, generalized to whatever neighborhood is chosen above.
# BIRTH_COUNTS: neighbor counts (of FULLY ALIVE neighbors only) that bring a dead
# cell to life. SURVIVE_COUNTS: neighbor counts that let a fully alive cell stay
# fully alive instead of starting to decay. These defaults reproduce the original
# hardcoded rule (survive on 6-8, born on exactly 5).
BIRTH_COUNTS = {5}
SURVIVE_COUNTS = {6, 7, 8}

# "Generations"-style states: 0 = dead/quiescent, NUM_STATES-1 = fully alive
# (just born or freshly surviving). Values in between are a forced refractory
# countdown -- a decaying cell always drops by exactly 1 state per frame
# regardless of its neighbor count, and cannot be reborn until it reaches 0.
# Only fully-alive cells (state == NUM_STATES-1) count as "alive" neighbors for
# other cells' birth/survival checks -- decaying cells are invisible to the rule,
# matching the standard Generations/Brian's-Brain convention.
# NUM_STATES = 2 (the default) has no refractory states at all and reproduces
# the original 2-state (boolean) behavior exactly.
NUM_STATES = 2
MAX_STATE = NUM_STATES - 1
assert NUM_STATES >= 2, "NUM_STATES must be at least 2 (0 = dead, 1 = alive)"

# ==========================================
# 1c. Rule String (Visions-of-Chaos "S/B/C/N" format)
# ==========================================
# If set, RULE_STRING OVERRIDES the manual SURVIVE/BIRTH/NUM_STATES/NEIGHBORHOOD
# variables above. Format: survival/birth/states/neighborhood, with commas
# separating counts and hyphens for inclusive ranges. "M" = Moore, "VN" = von
# Neumann. An empty section means "no counts". Examples:
#   "4/4/5/M"                  -> the classic 445 rule
#   "9-26/5-7,12-13,15/5/M"    -> Amoeba
#   "/4/2/M"                   -> no survival at all, birth on exactly 4
# RULE_STRING can also be the name of a preset from RULE_PRESETS below.
# Set to None to use the manual variables in section 1b instead.
RULE_STRING = "0,9,15-17,20/4,8,12,14,23/3/M"

# Named presets in VoC S/B/C/N format (all radius-1 Moore, as published on
# Softology's 3D Cellular Automata blog).
RULE_PRESETS = {
    "445":          "4/4/5/M",
    "amoeba":       "9-26/5-7,12-13,15/5/M",
    "architecture": "4-6/3/2/M",
    "clouds-1":     "13-26/13-14,17-19/2/M",
    "3d-brain":     "/4/2/M",
    "pyroclastic":  "4-7/6-8/10/M",
    "original":     "6-8/5/2/M",   # this script's original hardcoded rule
}

# ==========================================
# 1d. Initial Seed Configuration
# ==========================================
# How the grid is populated at frame 0 of each run:
#   "soup"         -> every cell in the grid random at PROB_ALIVE density (original behavior)
#   "center_block" -> a centered SEED_BLOCK_SIZE^3 region random at PROB_ALIVE density,
#                     everything else dead. Growth rules like 445 basically require this;
#                     with a full soup they tend to instantly die or explode.
#   "single"       -> one live cell at the exact center of the grid
#   "shell"        -> a centered hollow 1-cell-thick box shell of size SEED_BLOCK_SIZE,
#                     random at PROB_ALIVE density on the shell surface
SEED_MODE = "center_block"
SEED_BLOCK_SIZE = 50

# ==========================================
# 1e. Boundary Mode
# ==========================================
#   "wrap" -> torus topology; patterns exiting one face re-enter the opposite face
#             (original behavior)
#   "dead" -> bounded grid with dead walls; everything beyond the edge counts as
#             permanently empty. Expanding rules behave very differently here --
#             they die/flatten at the walls instead of colliding with themselves.
BOUNDARY_MODE = "wrap"

# ==========================================
# 1f. Rule Search / Mining Mode
# ==========================================
# When True, each run gets a RANDOMLY GENERATED rule (survive/birth/states) instead
# of the configured one. The rule is printed, logged in the end-of-batch summary,
# and embedded in each run's metadata, so interesting long-lived runs can be
# reproduced later. The existing termination detection acts as an automatic filter:
# extinctions and instant-stabilizers are the boring ones. Neighborhood type/radius
# stay as configured (not randomized). Fully reproducible via RANDOM_SEED.
RULE_SEARCH_MODE = False
RULE_SEARCH_STATES_RANGE = (2, 10)     # inclusive range for random NUM_STATES
RULE_SEARCH_MAX_CONDITIONS = 6         # max distinct counts drawn for each of S and B

# ==========================================
# 1h. Phase-2 Culler Selection
# ==========================================
# Which culler runs after the density threshold is crossed (phase 1 is always FLOOD,
# whose thickness-shrink + point cap is required for the dense early soup):
#   "basic"    -> face culling: keeps any occupied cell touching ANY empty cell.
#                 Fast single pass, but also keeps the walls of fully sealed
#                 internal cavities that can never be seen from outside.
#   "exterior" -> exterior-hull culling: flood-fills the air from the grid
#                 boundary (geodesic dilation), keeps only occupied cells touching
#                 OUTSIDE-connected air. The exact true hollow shape: sealed
#                 cavities culled, concave pockets/tunnels correctly kept.
#                 Strictly <= "basic" cell counts -> smaller files.
PHASE2_CULLER = "exterior"

# Warm-start iterations per frame for the exterior flood (it reuses the previous
# frame's exterior-air solution, which only needs a few dilation steps to track
# frame-to-frame changes; ~3 typically suffice, 6 leaves margin).
EXTERIOR_WARM_ITERS = 6
# Every N phase-2 frames, redo the flood from scratch to full convergence, purging
# any stale exterior marking of pockets that sealed shut. Errors between refreshes
# are tiny and strictly on the "show a few extra cells" side (never holes).
# 0 = never reconverge (first frame is always a full cold solve regardless).
EXTERIOR_RECONVERGE_EVERY = 100

# ==========================================
# 1g. Rule String Parsing & Formatting
# ==========================================
def parse_counts(section):
    """Parses one S or B section of a rule string: comma-separated numbers
    and inclusive hyphen ranges, e.g. '5-7,12-13,15' -> {5,6,7,12,13,15}.
    An empty section means no counts at all."""
    counts = set()
    section = section.strip()
    if not section:
        return counts
    for token in section.split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = token.split("-")
            lo, hi = int(lo), int(hi)
            if lo > hi:
                raise ValueError(f"Backwards range '{token}' in rule string")
            counts.update(range(lo, hi + 1))
        else:
            counts.add(int(token))
    return counts

def parse_rule_string(rule_str):
    """Parses a Visions-of-Chaos style 'S/B/C/N' rule string (or a RULE_PRESETS
    name) into (survive_counts, birth_counts, num_states, neighborhood_type)."""
    if rule_str in RULE_PRESETS:
        rule_str = RULE_PRESETS[rule_str]
    parts = rule_str.strip().split("/")
    if len(parts) != 4:
        raise ValueError(
            f"Rule string {rule_str!r} must have exactly 4 sections "
            f"(survival/birth/states/neighborhood), got {len(parts)}"
        )
    survive = parse_counts(parts[0])
    birth = parse_counts(parts[1])
    num_states = int(parts[2])
    n_token = parts[3].strip().upper()
    if n_token == "M":
        ntype = "moore"
    elif n_token in ("VN", "N"):
        ntype = "von_neumann"
    else:
        raise ValueError(f"Unknown neighborhood token {parts[3]!r} (expected 'M' or 'VN')")
    return survive, birth, num_states, ntype

def format_counts(counts):
    """Formats a set of counts back into compact 'a-b,c' range notation."""
    if not counts:
        return ""
    sorted_counts = sorted(counts)
    ranges = []
    start = prev = sorted_counts[0]
    for n in sorted_counts[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append((start, prev))
            start = prev = n
    ranges.append((start, prev))
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)

def format_rule_string(survive, birth, num_states, ntype):
    """Builds the canonical S/B/C/N string for the given rule components."""
    n_token = "M" if ntype == "moore" else "VN"
    return f"{format_counts(survive)}/{format_counts(birth)}/{num_states}/{n_token}"

# Apply RULE_STRING override (if configured) on top of the manual 1b variables
if RULE_STRING is not None:
    SURVIVE_COUNTS, BIRTH_COUNTS, NUM_STATES, NEIGHBORHOOD_TYPE = parse_rule_string(RULE_STRING)
    MAX_STATE = NUM_STATES - 1
    assert NUM_STATES >= 2, "NUM_STATES must be at least 2 (0 = dead, 1 = alive)"

# Canonical string for the currently active rule (updated per-run in search mode)
ACTIVE_RULE_STRING = format_rule_string(SURVIVE_COUNTS, BIRTH_COUNTS, NUM_STATES, NEIGHBORHOOD_TYPE)
print(f"[*] Active rule: {ACTIVE_RULE_STRING}" + (" (RULE SEARCH MODE: randomized per run)" if RULE_SEARCH_MODE else ""))

# Automatic CAPACITY_THRESHOLD calculation if explicitly set to False
if CAPACITY_THRESHOLD is False:
    total_volume = GRID_SIZE[0] * GRID_SIZE[1] * GRID_SIZE[2]
    inner_volume = (
        max(0, GRID_SIZE[0] - 2 * SHELL_THICKNESS) *
        max(0, GRID_SIZE[1] - 2 * SHELL_THICKNESS) *
        max(0, GRID_SIZE[2] - 2 * SHELL_THICKNESS)
    )
    CAPACITY_THRESHOLD = 1.0 - (inner_volume / total_volume)
    print(f"[*] CAPACITY_THRESHOLD set to False. Auto-calculated bounding limit: {CAPACITY_THRESHOLD:.4f} ({CAPACITY_THRESHOLD:.2%})")

# ==========================================
# 2. PyTorch CUDA Initialization
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Physics Engine Initialized on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    # Forces CUDA to select deterministic algorithms
    torch.backends.cudnn.deterministic = True
    print(f"[*] Deterministic mode active. Random seed locked to: {RANDOM_SEED}")

# Physics Kernel
def build_neighborhood_kernel(neighborhood_type, radius, dtype, device):
    """Builds a convolution kernel representing which offset cells count as
    neighbors. Values are 1.0 for included cells, 0.0 for excluded (and always
    0.0 at the center, which is never its own neighbor).
    "moore": every cell within Chebyshev distance <= radius (a full cube).
    "von_neumann": only cells within Manhattan distance <= radius (face-connected)."""
    size = 2 * radius + 1
    kernel = torch.zeros((1, 1, size, size, size), dtype=dtype, device=device)
    center = radius
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                if neighborhood_type == "moore":
                    include = True  # loop bounds already enforce Chebyshev distance <= radius
                elif neighborhood_type == "von_neumann":
                    include = (abs(dz) + abs(dy) + abs(dx)) <= radius
                else:
                    raise ValueError(f"Unknown NEIGHBORHOOD_TYPE: {neighborhood_type!r} (expected 'moore' or 'von_neumann')")
                if include:
                    kernel[0, 0, center + dz, center + dy, center + dx] = 1.0
    return kernel

kernel = build_neighborhood_kernel(NEIGHBORHOOD_TYPE, NEIGHBORHOOD_RADIUS, torch.bfloat16, device)
print(f"[*] Neighborhood: {NEIGHBORHOOD_TYPE} (radius={NEIGHBORHOOD_RADIUS}, {int(kernel.sum().item())} cells) | States: {NUM_STATES} (max_state={MAX_STATE})")

def calculate_next_gen_tensor(state_tensor):
    """state_tensor holds integer states: 0 = dead, MAX_STATE = fully alive,
    anything in between is a decaying refractory state. Only fully-alive cells
    count as neighbors for the birth/survival rule (standard Generations
    convention) -- decaying cells always count down by 1 regardless of what's
    around them, and can't be reborn until they reach 0."""
    alive_mask = (state_tensor == MAX_STATE)
    alive_float = alive_mask.to(kernel.dtype)

    if BOUNDARY_MODE == "wrap":
        padded = F.pad(alive_float, (NEIGHBORHOOD_RADIUS,) * 6, mode='circular')
    else:  # "dead": bounded grid, everything beyond the walls counts as empty
        padded = F.pad(alive_float, (NEIGHBORHOOD_RADIUS,) * 6, mode='constant', value=0.0)
    neighbor_counts = F.conv3d(padded, kernel)
    # bf16/fp16 sums of small integers are exact, but round defensively before
    # casting so float rounding noise can never shift a count across a boundary.
    neighbor_counts = torch.round(neighbor_counts).to(torch.int16)

    dead_mask = (state_tensor == 0)
    refractory_mask = (~dead_mask) & (~alive_mask)

    birth_trigger = torch.zeros_like(dead_mask)
    for n in BIRTH_COUNTS:
        birth_trigger |= (neighbor_counts == n)

    survive_trigger = torch.zeros_like(alive_mask)
    for n in SURVIVE_COUNTS:
        survive_trigger |= (neighbor_counts == n)

    new_state = state_tensor.clone()
    # Dead cells: reborn at MAX_STATE if the birth condition is met, else stay dead.
    new_state = torch.where(dead_mask & birth_trigger, torch.full_like(state_tensor, MAX_STATE), new_state)
    # Fully alive cells: stay at MAX_STATE if survival condition met, else start decaying.
    new_state = torch.where(alive_mask & ~survive_trigger, torch.full_like(state_tensor, MAX_STATE - 1), new_state)
    # Refractory cells: forced countdown, no rule can stop or reverse this.
    # torch.where evaluates BOTH branches at every position before selecting,
    # so "state - 1" also gets computed at dead cells where uint8 0-1 would
    # wrap to 255. Those positions are never selected by refractory_mask, but
    # clamping makes this safe-by-construction instead of safe-by-coincidence.
    new_state = torch.where(refractory_mask, torch.clamp(state_tensor, min=1) - 1, new_state)
    return new_state

assert BOUNDARY_MODE in ("wrap", "dead"), f"Unknown BOUNDARY_MODE: {BOUNDARY_MODE!r}"
assert SEED_MODE in ("soup", "center_block", "single", "shell"), f"Unknown SEED_MODE: {SEED_MODE!r}"
assert PHASE2_CULLER in ("basic", "exterior"), f"Unknown PHASE2_CULLER: {PHASE2_CULLER!r}"

# ==========================================
# 2b. Initial Seed Builder & Random Rule Generator
# ==========================================
def build_initial_alive_grid():
    """Builds the frame-0 boolean 'alive here' grid according to SEED_MODE.
    Uses np.random, so it's covered by the per-run seeding for reproducibility."""
    if SEED_MODE == "soup":
        return np.random.choice(
            [False, True], size=GRID_SIZE, p=[1 - PROB_ALIVE, PROB_ALIVE]
        )

    grid = np.zeros(GRID_SIZE, dtype=bool)
    center = tuple(d // 2 for d in GRID_SIZE)

    if SEED_MODE == "single":
        grid[center] = True
        return grid

    # Centered block bounds, clamped so a SEED_BLOCK_SIZE larger than an axis
    # just fills that whole axis instead of erroring.
    block = [min(SEED_BLOCK_SIZE, d) for d in GRID_SIZE]
    lo = [(d - b) // 2 for d, b in zip(GRID_SIZE, block)]
    hi = [l + b for l, b in zip(lo, block)]

    if SEED_MODE == "center_block":
        region_shape = tuple(b for b in block)
        grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = (
            np.random.random(region_shape) < PROB_ALIVE
        )
        return grid

    if SEED_MODE == "shell":
        # 1-cell-thick hollow box: fill the outer block, then carve out the interior.
        region = np.random.random(tuple(block)) < PROB_ALIVE
        if all(b > 2 for b in block):
            region[1:-1, 1:-1, 1:-1] = False
        grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = region
        return grid

def generate_random_rule(neighbor_count):
    """Draws a random Generations rule for RULE_SEARCH_MODE. Birth never includes
    0 (a B0 rule under wrap instantly floods the entire grid) and always has at
    least one count; survival may legitimately be empty (Brian's-Brain-style).
    Uses the random module, so it's covered by per-run seeding."""
    n_survive = random.randint(0, RULE_SEARCH_MAX_CONDITIONS)
    n_birth = random.randint(1, RULE_SEARCH_MAX_CONDITIONS)
    survive = set(random.sample(range(0, neighbor_count + 1), k=min(n_survive, neighbor_count + 1)))
    birth = set(random.sample(range(1, neighbor_count + 1), k=min(n_birth, neighbor_count)))
    num_states = random.randint(*RULE_SEARCH_STATES_RANGE)
    return survive, birth, num_states

def generate_run_graph(run_dir, metadata, population, visible_population,
                       fps_history, culler_modes, thickness_history,
                       delta_counts, wall_seconds):
    """Saves run_overview.png into the run folder: a full-run dashboard with
    populations + culler phases, FPS, and per-frame delta sizes, annotated
    with the complete run configuration. Never raises -- a plotting problem
    should not be able to kill a batch."""
    try:
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless backend; no display needed
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [!] matplotlib not installed -- skipping run graph. (pip install matplotlib)")
            return

        frames = np.arange(len(population))
        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.4, 1.6]})
        ax_pop, ax_fps, ax_delta = axes

        # ---- Panel 1: populations + culler phase annotations ----
        ax_pop.plot(frames, population, color="tab:blue", lw=1.5, label="Alive cells (pre-cull)")
        ax_pop.plot(frames, visible_population, color="tab:orange", lw=1.5, label="Visible cells (post-cull)")
        thr_count = metadata["capacity_threshold"] * np.prod(metadata["grid_size"])
        ax_pop.axhline(thr_count, color="tab:red", ls=":", lw=1.2,
                       label=f"Oscillator threshold ({metadata['capacity_threshold']:.2%} of grid)")

        threshold_frame = next((i for i, m in enumerate(culler_modes) if m != "FLOOD"), None)
        graduation_frame = next((i for i, m in enumerate(culler_modes) if "(caught up)" in m), None)
        if threshold_frame is not None:
            for ax in axes:
                ax.axvline(threshold_frame, color="tab:red", ls="--", lw=1.2)
            ax_pop.text(threshold_frame, ax_pop.get_ylim()[1] * 0.97, " threshold hit",
                        color="tab:red", fontsize=8, va="top")
        if graduation_frame is not None:
            for ax in axes:
                ax.axvline(graduation_frame, color="tab:green", ls="--", lw=1.2)
            ax_pop.text(graduation_frame, ax_pop.get_ylim()[1] * 0.90,
                        f" {metadata['phase2_culler'].upper()} active", color="tab:green",
                        fontsize=8, va="top")
            if threshold_frame is not None and graduation_frame > threshold_frame:
                for ax in axes:
                    ax.axvspan(threshold_frame, graduation_frame, color="tab:red", alpha=0.06)
        ax_pop.set_ylabel("Cells")
        ax_pop.legend(loc="upper right", fontsize=8)
        ax_pop.grid(alpha=0.25)

        # ---- Panel 2: FPS ----
        ax_fps.plot(frames, fps_history, color="tab:purple", lw=1.2)
        valid_fps = [f for f in fps_history if f == f and f > 0]  # drop NaN/zeros
        if valid_fps:
            avg = sum(valid_fps) / len(valid_fps)
            ax_fps.axhline(avg, color="tab:purple", ls=":", lw=1,
                           label=f"mean {avg:.1f} fps")
            ax_fps.legend(loc="upper right", fontsize=8)
        ax_fps.set_ylabel(f"FPS ({FPS_AVG_WINDOW}-frame avg)")
        ax_fps.grid(alpha=0.25)

        # ---- Panel 3: delta sizes + FLOOD thickness ----
        ax_delta.plot(frames, delta_counts, color="tab:brown", lw=1.2,
                      label="Delta entries / frame")
        ax_delta.set_ylabel("Delta entries")
        ax_delta.set_xlabel("Simulation frame")
        ax_delta.grid(alpha=0.25)
        thick = [t if t is not None else float("nan") for t in thickness_history]
        if any(t == t for t in thick):
            ax_t = ax_delta.twinx()
            ax_t.step(frames, thick, color="tab:gray", lw=1, where="mid", alpha=0.8,
                      label="FLOOD thickness")
            ax_t.set_ylabel("FLOOD thickness", color="tab:gray")
            ax_t.set_ylim(0, SHELL_THICKNESS + 1)
        ax_delta.legend(loc="upper right", fontsize=8)

        # ---- Title + config summary ----
        m = metadata
        fig.suptitle(
            f"Run {m['run_index']:03d}  |  Rule {m['rule_string']}  |  "
            f"Grid {'x'.join(map(str, m['grid_size']))}  |  {m['num_states']} states  |  "
            f"{m['neighborhood_type']} r={m['neighborhood_radius']}  |  boundary={m['boundary_mode']}",
            fontsize=11)
        start_cells = population[0] if len(population) else 0
        seed_desc = m["seed_mode"]
        if m["seed_mode"] in ("center_block", "shell"):
            seed_desc += f" (size {m['seed_block_size']})"
        summary = (
            f"Seed: {seed_desc} @ prob {m['prob_alive']}  |  run_seed={m['run_seed']}  |  "
            f"start cells: {start_cells:,}  |  peak alive: {max(population):,}  |  "
            f"peak visible: {max(visible_population):,}\n"
            f"{m['total_frames']} frames in {wall_seconds:.1f}s  |  "
            f"cullers: FLOOD (diag={m['flood_diagonals']}) "
            f"-> {m['phase2_culler']}  |  outcome: {m['termination_reason']}"
        )
        fig.text(0.5, 0.955, summary, ha="center", va="top", fontsize=8.5)
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        out_path = os.path.join(run_dir, "run_overview.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  [i] Run overview graph saved: {out_path}")
    except Exception as e:
        print(f"  [!] Run graph generation failed (batch unaffected): {e}")

# ==========================================
# 3. CULLING FUNCTIONS (Now returning Boolean Masks)
# ==========================================
def _shear(A):
    """Diagonal-to-column shear on the last two dims: returns S with
    S[..., a, b] = A[..., a, b-a] (zeros where out of range). Implemented as
    pad + reflow: padding each row by H zeros then reinterpreting the flat
    buffer at width K-1 shifts every row left by its own row index -- no
    gather ops needed. Same trick inverted in _unshear."""
    H, W = A.shape[-2], A.shape[-1]
    K = W + H
    pad_shape = A.shape[:-1] + (K - W,)
    P = torch.cat([A, torch.zeros(pad_shape, dtype=A.dtype, device=A.device)], dim=-1)
    flat = P.reshape(*A.shape[:-2], H * K)
    return flat[..., :H * (K - 1)].reshape(*A.shape[:-2], H, K - 1)

def _unshear(S, W):
    """Inverse of _shear: returns U with U[..., a, x] = S[..., a, x+a],
    sliced back to original width W."""
    H, K1 = S.shape[-2], S.shape[-1]
    flat = S.reshape(*S.shape[:-2], H * K1)
    pad_shape = S.shape[:-2] + (H,)
    flat = torch.cat([flat, torch.zeros(pad_shape, dtype=S.dtype, device=S.device)], dim=-1)
    return flat.reshape(*S.shape[:-2], H, K1 + 1)[..., :W]

def _diag_cumsums_plane(grid_int_3d, axis_a, axis_b):
    """Inclusive occupied-count along all 4 diagonal ray directions in the
    (axis_a, axis_b) plane of a 3D tensor -- the diagonal analog of the six
    axis cumsums. Shear couples the two axes with OPPOSITE signs (S[a,b] =
    A[a,b-a] walks left as it walks down), so cumsum direction sa on shear
    layout sb computes the ray traveling (sa, -sa*sb); results are returned
    as a list under that corrected direction (validated against brute-force
    ray marching in all 12 directions)."""
    order = [d for d in range(3) if d not in (axis_a, axis_b)] + [axis_a, axis_b]
    inv = [order.index(d) for d in range(3)]
    T = grid_int_3d.permute(order).contiguous()
    W = T.shape[-1]
    results = []
    for sb in (1, -1):
        Tb = T if sb == 1 else torch.flip(T, dims=[-1])
        S = _shear(Tb)
        for sa in (1, -1):
            Sa = S if sa == 1 else torch.flip(S, dims=[-2])
            C = torch.cumsum(Sa, dim=-2)
            C = C if sa == 1 else torch.flip(C, dims=[-2])
            U = _unshear(C, W)
            U = U if sb == 1 else torch.flip(U, dims=[-1])
            results.append(U.permute(inv))
    return results

def get_visible_hull_mask(occupied_mask):
    """occupied_mask: boolean tensor, True wherever a cell is in ANY non-zero
    state (fully alive or decaying) -- this culls based on occupancy in space,
    not on which specific state a cell is in."""
    grid_int = occupied_mask.to(torch.int16)

    # These cumulative sums don't depend on current_thickness at all -- only
    # the "<= current_thickness" comparison below does. Computing them once
    # and reusing across candidate thicknesses avoids redoing 6 full
    # cumsum+flip passes for every thickness the loop tries (up to
    # SHELL_THICKNESS times per frame).
    z_cum_fwd = torch.cumsum(grid_int, dim=2)
    z_cum_bwd = torch.flip(torch.cumsum(torch.flip(grid_int, dims=[2]), dim=2), dims=[2])
    y_cum_fwd = torch.cumsum(grid_int, dim=3)
    y_cum_bwd = torch.flip(torch.cumsum(torch.flip(grid_int, dims=[3]), dim=3), dims=[3])
    x_cum_fwd = torch.cumsum(grid_int, dim=4)
    x_cum_bwd = torch.flip(torch.cumsum(torch.flip(grid_int, dims=[4]), dim=4), dims=[4])

    # Optional 12 edge-diagonal ray counts (45-degree directions), hoisted out
    # of the thickness loop exactly like the axis cumsums above.
    diag_counts = []
    if FLOOD_DIAGONALS:
        grid_int_3d = grid_int.squeeze()
        for (a, b) in ((0, 1), (0, 2), (1, 2)):
            diag_counts.extend(_diag_cumsums_plane(grid_int_3d, a, b))
        diag_counts = [c.unsqueeze(0).unsqueeze(0) for c in diag_counts]

    current_thickness = SHELL_THICKNESS
    while current_thickness >= 1:
        z_first = occupied_mask & (z_cum_fwd <= current_thickness)
        z_last  = occupied_mask & (z_cum_bwd <= current_thickness)
        y_first = occupied_mask & (y_cum_fwd <= current_thickness)
        y_last  = occupied_mask & (y_cum_bwd <= current_thickness)
        x_first = occupied_mask & (x_cum_fwd <= current_thickness)
        x_last  = occupied_mask & (x_cum_bwd <= current_thickness)

        shell_mask = z_first | z_last | y_first | y_last | x_first | x_last

        for dc in diag_counts:
            shell_mask |= occupied_mask & (dc <= current_thickness)

        if not MAX_PHASE1_POINTS or current_thickness == 1:
            break

        if shell_mask.sum().item() <= MAX_PHASE1_POINTS:
            break

        current_thickness -= 1

    # Calculate absolute maximum cells that can exist at this specific thickness
    total_vol = GRID_SIZE[0] * GRID_SIZE[1] * GRID_SIZE[2]
    inner_vol = (
        max(0, GRID_SIZE[0] - 2 * current_thickness) *
        max(0, GRID_SIZE[1] - 2 * current_thickness) *
        max(0, GRID_SIZE[2] - 2 * current_thickness)
    )
    max_cells = total_vol - inner_vol

    return shell_mask.squeeze(), current_thickness, max_cells # Return data tuple

def get_basic_surface_mask(occupied_mask):
    """occupied_mask: boolean tensor, True wherever a cell is in ANY non-zero
    state. Returns cells on the outer surface of the occupied volume."""
    padded = F.pad(occupied_mask, (1, 1, 1, 1, 1, 1), mode='constant', value=False)
    inv_padded = ~padded

    has_empty_neighbor = torch.zeros_like(padded, dtype=torch.bool)
    has_empty_neighbor[:, :, 1:, :, :] |= inv_padded[:, :, :-1, :, :]
    has_empty_neighbor[:, :, :-1, :, :] |= inv_padded[:, :, 1:, :, :]
    has_empty_neighbor[:, :, :, 1:, :] |= inv_padded[:, :, :, :-1, :]
    has_empty_neighbor[:, :, :, :-1, :] |= inv_padded[:, :, :, 1:, :]
    has_empty_neighbor[:, :, :, :, 1:] |= inv_padded[:, :, :, :, :-1]
    has_empty_neighbor[:, :, :, :, :-1] |= inv_padded[:, :, :, :, 1:]

    surface_mask = padded & has_empty_neighbor
    return surface_mask[:, :, 1:-1, 1:-1, 1:-1].squeeze()

def _dilate6(x):
    """One step of 6-connected (face-neighbor) boolean dilation on a 5D tensor.
    Pure boolean slice-ORs -- same op pattern as get_basic_surface_mask, no
    float casts, no wrap (grid edges act as walls, which is what we want:
    the camera is outside the box regardless of the sim's BOUNDARY_MODE)."""
    out = x.clone()
    out[:, :, 1:, :, :]  |= x[:, :, :-1, :, :]
    out[:, :, :-1, :, :] |= x[:, :, 1:, :, :]
    out[:, :, :, 1:, :]  |= x[:, :, :, :-1, :]
    out[:, :, :, :-1, :] |= x[:, :, :, 1:, :]
    out[:, :, :, :, 1:]  |= x[:, :, :, :, :-1]
    out[:, :, :, :, :-1] |= x[:, :, :, :, 1:]
    return out

# Warm-start state for the exterior flood; reset at the start of every run.
_exterior_air_prev = None
_exterior_frame_counter = 0

def get_exterior_hull_mask(occupied_mask):
    """Exterior-hull culler: flood-fills empty space from the grid boundary
    (geodesic dilation: grow the front 1 voxel per iteration, occupied cells
    block it), then keeps occupied cells face-adjacent to that OUTSIDE-connected
    air. Sealed internal cavities are never reached -> their walls are culled;
    open pockets/tunnels are reached -> their walls are kept. This is the exact
    visible hull for face-adjacency visibility.

    Cost control: the first call (and every EXTERIOR_RECONVERGE_EVERY frames)
    solves from scratch to full convergence. All other frames warm-start from
    the previous frame's exterior air, which only needs EXTERIOR_WARM_ITERS
    dilation steps to track frame-to-frame changes. Warm-start error is tiny,
    transient, and strictly on the over-show side (a pocket that seals shut
    stays marked exterior until the next full reconvergence -- extra cells
    shown, never holes)."""
    global _exterior_air_prev, _exterior_frame_counter

    air = ~occupied_mask

    # Seed: air on the six boundary faces of the grid.
    seed = torch.zeros_like(air)
    seed[:, :, 0, :, :]  = air[:, :, 0, :, :]
    seed[:, :, -1, :, :] = air[:, :, -1, :, :]
    seed[:, :, :, 0, :]  = air[:, :, :, 0, :]
    seed[:, :, :, -1, :] = air[:, :, :, -1, :]
    seed[:, :, :, :, 0]  = air[:, :, :, :, 0]
    seed[:, :, :, :, -1] = air[:, :, :, :, -1]

    do_full_solve = (
        _exterior_air_prev is None
        or (EXTERIOR_RECONVERGE_EVERY and _exterior_frame_counter % EXTERIOR_RECONVERGE_EVERY == 0)
    )

    if do_full_solve:
        # Cold: iterate to convergence. The equality check syncs the GPU each
        # iteration, but full solves are rare (first frame + periodic refresh).
        ext_air = seed
        while True:
            grown = _dilate6(ext_air) & air
            if torch.equal(grown, ext_air):
                break
            ext_air = grown
    else:
        # Warm: previous exterior air (minus anything now occupied) is already
        # ~99.9% of the answer; a few dilation steps track this frame's changes.
        ext_air = (_exterior_air_prev & air) | seed
        for _ in range(EXTERIOR_WARM_ITERS):
            ext_air = _dilate6(ext_air) & air

    _exterior_air_prev = ext_air
    _exterior_frame_counter += 1

    # Visible = occupied cells with an exterior-air face neighbor. _dilate6
    # includes ext_air itself, but ext_air & occupied is empty by construction.
    return (occupied_mask & _dilate6(ext_air)).squeeze()

def get_phase2_mask(occupied_mask):
    """Dispatches to the configured phase-2 culler."""
    if PHASE2_CULLER == "exterior":
        return get_exterior_hull_mask(occupied_mask)
    return get_basic_surface_mask(occupied_mask)

def split_and_cast_coords_and_states(tensor_coords, tensor_states):
    """Splits a coordinate tensor into uint8/uint16 numpy arrays by magnitude
    (same as before), AND splits an aligned per-cell state array using the
    SAME row mask, so coords and states stay correctly paired by row index
    within each precision tier. tensor_states holds the new state value
    (0-MAX_STATE) for each row in tensor_coords; 0 means the cell should be
    treated as removed/hidden."""
    if tensor_coords.numel() == 0:
        empty_c8 = np.empty((0, 3), dtype=np.uint8)
        empty_c16 = np.empty((0, 3), dtype=np.uint16)
        empty_s = np.empty((0,), dtype=np.uint8)
        return empty_c8, empty_c16, empty_s, empty_s

    # Find points where the maximum XYZ value is strictly less than 256
    max_vals, _ = torch.max(tensor_coords, dim=1)
    mask_8bit = max_vals < 256

    coords_8 = tensor_coords[mask_8bit].cpu().numpy().astype(np.uint8)
    coords_16 = tensor_coords[~mask_8bit].cpu().numpy().astype(np.uint16)

    states_8 = tensor_states[mask_8bit].cpu().numpy().astype(np.uint8)
    states_16 = tensor_states[~mask_8bit].cpu().numpy().astype(np.uint8)

    return coords_8, coords_16, states_8, states_16

# ==========================================
# 4. Decoupled I/O Worker (Upgraded for robustness & large files)
# ==========================================
MAX_CACHE_SIZE = 2
frame_cache = queue.Queue(maxsize=MAX_CACHE_SIZE)

# Threads used to compress the arrays WITHIN a single run's zip, in parallel.
# zlib.compressobj releases the GIL during compress()/flush(), so this gets
# genuine multi-core speedup rather than fighting the GIL. Since NUM_WORKERS
# runs can also be zipping concurrently, up to NUM_WORKERS * this many
# threads may be active on the system at once (fine per current setup).
COMPRESSION_THREADS_PER_FILE = 16

def _compress_entry(arcname, arr, level=6):
    """Serializes an array and deflate-compresses it. Safe to call from
    multiple threads concurrently -- each call only touches its own buffer."""
    buf = io.BytesIO()
    np.save(buf, arr)
    raw = buf.getvalue()
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    compressor = zlib.compressobj(level, zlib.DEFLATED, -15)  # raw deflate, no zlib/gzip wrapper
    compressed = compressor.compress(raw) + compressor.flush()
    return arcname, compressed, crc, len(raw)

def _write_zip_manual(filename, entries):
    """Assembles a standard ZIP (readable by np.load/zipfile) from entries
    that are ALREADY compressed: (arcname, compressed_bytes, crc32, raw_size).
    zipfile.ZipFile has no supported way to accept pre-compressed data for a
    ZIP_DEFLATED entry, so this bypasses it -- the actual compression work
    already happened in parallel; this just writes bytes sequentially, which
    is cheap since there's no CPU-bound work left to do here."""
    central_directory = []
    offset = 0

    with open(filename, 'wb') as f:
        for arcname, comp_data, crc, raw_size in entries:
            name_bytes = arcname.encode('utf-8')
            comp_size = len(comp_data)

            local_header = struct.pack(
                '<IHHHHHIIIHH',
                0x04034b50, 20, 0, 8, 0, 0x21,
                crc, comp_size, raw_size, len(name_bytes), 0
            )
            f.write(local_header)
            f.write(name_bytes)
            f.write(comp_data)

            central_directory.append((name_bytes, crc, comp_size, raw_size, offset))
            offset += len(local_header) + len(name_bytes) + comp_size

        cd_start = offset
        for name_bytes, crc, comp_size, raw_size, local_offset in central_directory:
            central_header = struct.pack(
                '<IHHHHHHIIIHHHHHII',
                0x02014b50, 20, 20, 0, 8, 0, 0x21,
                crc, comp_size, raw_size, len(name_bytes), 0, 0, 0, 0, 0,
                local_offset
            )
            f.write(central_header)
            f.write(name_bytes)

        cd_size = f.tell() - cd_start
        eocd = struct.pack(
            '<IHHHHIIH',
            0x06054b50, 0, 0, len(central_directory), len(central_directory),
            cd_size, cd_start, 0
        )
        f.write(eocd)

def file_io_worker():
    while True:
        task = frame_cache.get()
        if task is None:
            break

        run_dict, filename = task
        try:
            # Compress every array in this run in parallel (multi-threaded,
            # bypasses the single-thread zlib limit), then assemble the zip
            # container sequentially with the pre-compressed bytes.
            with ThreadPoolExecutor(max_workers=COMPRESSION_THREADS_PER_FILE) as executor:
                futures = [
                    executor.submit(_compress_entry, f"{key}.npy", arr)
                    for key, arr in run_dict.items()
                ]
                entries = [fut.result() for fut in futures]

            _write_zip_manual(filename, entries)

            # This run's arrays are now on disk and every reference to them
            # (task, run_dict, entries, futures) is about to go out of scope.
            # Ask glibc to actually hand the freed memory back to the OS
            # instead of silently retaining it in the heap for reuse.
            del task, run_dict, entries, futures
            _release_freed_memory_to_os()
        except Exception as e:
            print(f"\n[ERROR] Background worker failed saving {filename}: {e}")
        finally:
            # Essential: ensure task_done is always hit so the main thread never deadlocks
            frame_cache.task_done()

NUM_WORKERS = 4
workers = []
print(f"Starting {NUM_WORKERS} background I/O workers for delta-compression...")
for _ in range(NUM_WORKERS):
    t = threading.Thread(target=file_io_worker)
    t.start()
    workers.append(t)

# ==========================================
# 5. Main Batch Processing Loop
# ==========================================
_ = calculate_next_gen_tensor(torch.zeros((1, 1, 5, 5, 5), dtype=torch.uint8, device=device))

batch_start_time = time.time()
successful_runs_log = []
dead_run_dirs = []


for run_idx in range(1, TOTAL_SIMULATION_RUNS + 1):
    run_dir = os.path.join(BASE_OUTPUT_DIR, f"run_{run_idx:03d}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    print(f"\n==================================================")
    print(f" STARTING SIMULATION RUN {run_idx}/{TOTAL_SIMULATION_RUNS}")
    print(f"==================================================")

    current_run_seed = None
    if RANDOM_SEED is not None:
        current_run_seed = RANDOM_SEED + run_idx
        random.seed(current_run_seed)
        np.random.seed(current_run_seed)
        torch.manual_seed(current_run_seed)
        torch.cuda.manual_seed_all(current_run_seed)

    # Rule search mode: draw a fresh random rule for this run. Plain assignment
    # rebinds the module-level globals that calculate_next_gen_tensor reads at
    # call time. Neighborhood (and therefore the conv kernel) stays fixed.
    if RULE_SEARCH_MODE:
        neighbor_count = int(kernel.sum().item())
        SURVIVE_COUNTS, BIRTH_COUNTS, NUM_STATES = generate_random_rule(neighbor_count)
        MAX_STATE = NUM_STATES - 1
        ACTIVE_RULE_STRING = format_rule_string(SURVIVE_COUNTS, BIRTH_COUNTS, NUM_STATES, NEIGHBORHOOD_TYPE)
        print(f"  [i] Rule for this run: {ACTIVE_RULE_STRING}")

    np_grid_alive = build_initial_alive_grid()
    # Freshly alive cells start at MAX_STATE; everything else starts dead (0).
    np_state = np.where(np_grid_alive, MAX_STATE, 0).astype(np.uint8)

    state_tensor = torch.tensor(np_state, device=device).unsqueeze(0).unsqueeze(0)

    total_volume = state_tensor.numel()
    below_threshold = False          # density state, re-evaluated EVERY frame (not a latch)
    basic_culler_ready = False
    pre_switch_baseline_remaining = None
    history_counts = {}

    frame_idx = 0
    termination_reason = "Reached Max Frames"

    run_data = {}
    last_visible_state = None # Per-cell visible state (0 = hidden) from the previous frame
    # Reset the exterior culler's warm-start cache: a new run is brand-new
    # geometry, so the first phase-2 frame must do a full cold solve.
    _exterior_air_prev = None
    _exterior_frame_counter = 0
    population_history = []          # occupied (any non-zero state) cells per frame
    visible_population_history = []  # post-culling visible cells per frame
    frame_timestamps = []            # wall-clock time at the start of each frame
    fps_history = []                 # windowed frames/s per frame (nan until 2 frames exist)
    culler_mode_history = []         # which culler produced each frame
    thickness_history = []           # FLOOD thickness used per frame (None in phase 2)
    delta_count_history = []         # coordinate entries written per frame (file-size proxy)
    run_start_time = time.time()

    while True:
        if MAX_FRAMES > 0 and frame_idx >= MAX_FRAMES:
            break

        # NOTE: state_tensor holds integer STATE values now (0..MAX_STATE), not
        # booleans -- summing it directly would sum state values, not count
        # occupied cells, so we explicitly count cells with any non-zero state.
        current_live_cells = (state_tensor > 0).sum().item()

        # Density is evaluated EVERY frame -- not a one-way latch. Growth rules
        # (e.g. amoeba) can start sparse and explode back to soup density, and
        # runs can start already below the threshold (non-soup seeds, or an
        # auto-threshold above the initial fill). The culler choice must follow
        # the grid's actual state in both directions.
        current_capacity = current_live_cells / total_volume
        was_below = below_threshold
        below_threshold = current_capacity <= CAPACITY_THRESHOLD

        if below_threshold and not was_below:
            if pre_switch_baseline_remaining is None:
                print(f"\n  [+] Below threshold from the start ({current_capacity:.2%}); no FLOOD baseline exists -- {PHASE2_CULLER.upper()} culler trusted immediately. Oscillator tracking active.")
            else:
                print(f"\n  [+] Threshold hit ({current_capacity:.2%}). Oscillator tracking active; validating {PHASE2_CULLER.upper()} culler against FLOOD baseline before switching.")
        elif was_below and not below_threshold:
            # Grid re-densified above the threshold: FLOOD (and its point cap)
            # takes back over. The phase-2 validation state is reset so the
            # next descent revalidates against a fresh FLOOD baseline, and the
            # exterior culler's warm cache is dropped (it would be badly stale
            # by the time phase 2 re-engages).
            basic_culler_ready = False
            pre_switch_baseline_remaining = None
            _exterior_air_prev = None
            print(f"\n  [+] Grid re-densified above threshold ({current_capacity:.2%}); reverting to FLOOD culling.")

        if below_threshold:
            # Hashing the full integer state (not just occupancy) is required
            # for correct oscillator detection with multi-state rules -- two
            # configurations with the same occupied cells but different
            # refractory states are NOT the same state. Only hashed below the
            # threshold: dense chaotic grids never repeat, and the full-grid
            # GPU->CPU transfer is wasted there.
            cpu_bytes = state_tensor.cpu().numpy().tobytes()
            grid_hash = hash(cpu_bytes)
            history_counts[grid_hash] = history_counts.get(grid_hash, 0) + 1

            if history_counts[grid_hash] >= REPEAT_CUTOFF:
                termination_reason = f"Stabilized (Oscillator repeated {REPEAT_CUTOFF}x)"
                print(f"\n  [!] Stable state detected.")
                break

        if current_live_cells == 0:
            termination_reason = "Total Extinction"
            print(f"\n  [!] Grid died out completely.")
            break

        # ---------------------------------------------
        # BLAZING FAST GPU DELTA CALCULATION
        # ---------------------------------------------

        # Culling operates on OCCUPANCY (any non-zero state), not on which
        # specific state a cell is in -- a decaying refractory cell still
        # takes up space and should still be considered for the visible surface.
        occupied_mask = state_tensor > 0

        # 1. Get the pure boolean mask of visible cells and generate dynamic stats
        if not below_threshold:
            current_mask, used_thickness, max_cells = get_visible_hull_mask(occupied_mask)
            remaining_cells = current_mask.sum().item()
            # Keeps getting overwritten every FLOOD frame, so this naturally
            # freezes at "the remaining count right before the grid dropped
            # below the threshold" the instant the crossing happens.
            pre_switch_baseline_remaining = remaining_cells
            culler_mode = "FLOOD"
            stats_str = f"Thickness: {used_thickness:<7} | Max Cells: {max_cells:<7} | Remaining: {remaining_cells:<7} | Difference: {current_live_cells - remaining_cells:<7}"

        elif basic_culler_ready:
            current_mask = get_phase2_mask(occupied_mask)
            remaining_cells = current_mask.sum().item()
            culler_mode = PHASE2_CULLER.upper()
            stats_str = f"Remaining: {remaining_cells:<7} | Difference: {current_live_cells - remaining_cells:<7}"

        else:
            # Just crossed below the density threshold. The phase-2 culler tends
            # to report a much higher remaining count than FLOOD was reporting
            # right before the switch, which looks like a sudden jump in visible
            # geometry -- so don't trust it until it can match (or beat) FLOOD's
            # last remaining count; until then, redo the frame with FLOOD.
            # If NO baseline exists (the run has been below the threshold since
            # frame 0, so FLOOD never ran), there was no handoff and there is no
            # jump to guard against: trust the phase-2 culler immediately.
            candidate_mask = get_phase2_mask(occupied_mask)
            candidate_remaining = candidate_mask.sum().item()

            if pre_switch_baseline_remaining is None:
                basic_culler_ready = True
                current_mask = candidate_mask
                remaining_cells = candidate_remaining
                culler_mode = PHASE2_CULLER.upper()
                stats_str = f"Remaining: {remaining_cells:<7} | Difference: {current_live_cells - remaining_cells:<7}"
            elif candidate_remaining <= pre_switch_baseline_remaining:
                basic_culler_ready = True
                current_mask = candidate_mask
                remaining_cells = candidate_remaining
                culler_mode = f"{PHASE2_CULLER.upper()} (caught up)"
                stats_str = f"Remaining: {remaining_cells:<7} | Baseline: {pre_switch_baseline_remaining:<7} | Difference: {current_live_cells - remaining_cells:<7}"
            else:
                current_mask, used_thickness, max_cells = get_visible_hull_mask(occupied_mask)
                remaining_cells = current_mask.sum().item()
                culler_mode = "FLOOD (fallback)"
                stats_str = f"Thickness: {used_thickness:<7} | Max Cells: {max_cells:<7} | Remaining: {remaining_cells:<7} | Baseline: {pre_switch_baseline_remaining:<7} | Difference: {current_live_cells - remaining_cells:<7}"

        # Per-frame stats history (feature: population graphs without loading in Blender)
        population_history.append(current_live_cells)
        visible_population_history.append(remaining_cells)
        culler_mode_history.append(culler_mode)
        thickness_history.append(used_thickness if culler_mode.startswith("FLOOD") else None)

        # Frame timing: windowed average over the last FPS_AVG_WINDOW frames.
        # Appended here (not at loop top) so a mid-frame termination break can
        # never leave the history arrays at mismatched lengths.
        frame_timestamps.append(time.time())
        if len(frame_timestamps) >= 2:
            w = frame_timestamps[-(FPS_AVG_WINDOW + 1):]
            current_fps = (len(w) - 1) / max(w[-1] - w[0], 1e-9)
        else:
            current_fps = float("nan")
        fps_history.append(current_fps)
        fps_str = f"{current_fps:6.1f}" if current_fps == current_fps else "   ---"

        # Real-time console update placed down here to catch calculated metrics
        print(f"\r  -> Frame {frame_idx:04d} | FPS: {fps_str} | Live Cells: {current_live_cells:<7} | Culler: {culler_mode} | {stats_str} | Zip Queue: {frame_cache.qsize()}/{MAX_CACHE_SIZE}", end="")

        # 2. Extract initial frame or calculate state-change deltas.
        #
        # NEW FILE FORMAT (state-aware, replaces the old born/died arrays):
        #   f_0_initial_8 / f_0_initial_16 : (N,3) coords of every visible cell in frame 0
        #   f_0_states_8  / f_0_states_16  : (N,)  uint8 state value of each of those cells,
        #                                    row-aligned with the coord array of the same tier
        #   f_{n}_delta_8 / f_{n}_delta_16 : (N,3) coords whose VISIBLE STATE changed vs frame n-1
        #   f_{n}_states_8 / f_{n}_states_16 : (N,) uint8 NEW state value for each coord.
        #                                    0 means the cell is now hidden/dead (remove it);
        #                                    any non-zero value means show it at that state.
        #
        # "Visible state" = the cell's state value where the culling mask is True, 0
        # everywhere else. Diffing visible states (instead of just the mask) captures
        # refractory countdowns on visible cells, so the loader can color/shade by state.
        state_3d = state_tensor.squeeze()
        visible_state = torch.where(current_mask, state_3d, torch.zeros_like(state_3d))

        if frame_idx == 0:
            coords = torch.nonzero(visible_state).flip(dims=[1])
            # nonzero() and boolean indexing traverse in the same row-major order,
            # so these state values stay row-aligned with the coords.
            cell_states = visible_state[visible_state > 0]

            c_8, c_16, s_8, s_16 = split_and_cast_coords_and_states(coords, cell_states)

            run_data["f_0_initial_8"] = c_8
            run_data["f_0_initial_16"] = c_16
            run_data["f_0_states_8"] = s_8
            run_data["f_0_states_16"] = s_16
        else:
            changed_mask = visible_state != last_visible_state

            changed_coords = torch.nonzero(changed_mask).flip(dims=[1])
            new_states = visible_state[changed_mask]

            c_8, c_16, s_8, s_16 = split_and_cast_coords_and_states(changed_coords, new_states)

            run_data[f"f_{frame_idx}_delta_8"] = c_8
            run_data[f"f_{frame_idx}_delta_16"] = c_16
            run_data[f"f_{frame_idx}_states_8"] = s_8
            run_data[f"f_{frame_idx}_states_16"] = s_16

        # Save current visible state for the next frame's delta comparison
        last_visible_state = visible_state
        delta_count_history.append(c_8.shape[0] + c_16.shape[0])
        # ---------------------------------------------

        state_tensor = calculate_next_gen_tensor(state_tensor)
        frame_idx += 1

    print(f"\n  [i] Run {run_idx} finished at frame {frame_idx}. Reason: {termination_reason}")

    # Self-documenting archive: embed everything needed to reproduce or
    # auto-configure this run. Stored as a 0-d numpy unicode array so np.load
    # can read it without allow_pickle; recover with:
    #   meta = json.loads(str(np.load(path)["metadata"]))
    # Keys don't start with "f_" and aren't (N,3)-shaped, so the loader's
    # existing frame/grid-size scans skip them automatically.
    metadata = {
        "format": "generations-delta-v1",
        "rule_string": ACTIVE_RULE_STRING,
        "survive_counts": sorted(SURVIVE_COUNTS),
        "birth_counts": sorted(BIRTH_COUNTS),
        "num_states": NUM_STATES,
        "neighborhood_type": NEIGHBORHOOD_TYPE,
        "neighborhood_radius": NEIGHBORHOOD_RADIUS,
        "boundary_mode": BOUNDARY_MODE,
        "phase2_culler": PHASE2_CULLER,
        "flood_diagonals": FLOOD_DIAGONALS,
        "grid_size": list(GRID_SIZE),
        "seed_mode": SEED_MODE,
        "seed_block_size": SEED_BLOCK_SIZE,
        "prob_alive": PROB_ALIVE,
        "run_seed": current_run_seed,
        "run_index": run_idx,
        "total_frames": frame_idx,
        "termination_reason": termination_reason,
        "shell_thickness": SHELL_THICKNESS,
        "capacity_threshold": CAPACITY_THRESHOLD,
    }
    run_data["metadata"] = np.array(json.dumps(metadata))
    run_data["population"] = np.array(population_history, dtype=np.int64)
    run_data["visible_population"] = np.array(visible_population_history, dtype=np.int64)

    if termination_reason == "Total Extinction" and DELETE_EXTINCTION_RUNS:
        dead_run_dirs.append(run_dir)
    else:
        if GENERATE_RUN_GRAPH:
            generate_run_graph(
                run_dir, metadata, population_history, visible_population_history,
                fps_history, culler_mode_history, thickness_history,
                delta_count_history, time.time() - run_start_time,
            )

        npz_filename = os.path.join(run_dir, f"data.npz")
        try:
            frame_cache.put((run_data, npz_filename), block=True)
        except queue.Full:
            pass

        successful_runs_log.append((run_idx, frame_idx, termination_reason, ACTIVE_RULE_STRING))

# ==========================================
# 6. Global Safe Shutdown
# ==========================================
print(f"\n==================================================")
print("ALL SIMULATIONS COMPLETE. Compressing remaining runs to SSD...")
print(f"==================================================")

initial_drain_size = frame_cache.qsize()
while not frame_cache.empty():
    print(f"\r  -> Remaining runs to zip: {frame_cache.qsize():<4} / {initial_drain_size:<4}", end="", flush=True)
    time.sleep(0.5)
print(f"\r  -> Remaining runs to zip: 0    / {initial_drain_size:<4} [Done!]")

frame_cache.join()

if DELETE_EXTINCTION_RUNS and dead_run_dirs:
    print(f"\nCleaning up {len(dead_run_dirs)} dead run directories from disk...")
    for folder in dead_run_dirs:
        if os.path.exists(folder):
            shutil.rmtree(folder)

print(f"\n==================================================")
print(f"BATCH COMPLETION SUMMARY")
print(f"==================================================")
print(f"Total processing time: {time.time() - batch_start_time:.2f} seconds.")
print(f"Saved {len(successful_runs_log)} compressed simulations to look at:")
for r_id, f_count, reason, rule_str in successful_runs_log:
    print(f"  -> Run {r_id:03d}: Lasted {f_count} frames | Rule: {rule_str} | Outcome: {reason}")

for _ in range(NUM_WORKERS):
    frame_cache.put(None)
for t in workers:
    t.join()

print("\nAll background processes terminated safely. Happy rendering!")
