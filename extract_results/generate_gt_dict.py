import json
import torch
from pathlib import Path
from itertools import combinations
from tqdm.auto import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/raid/create.nm43gr/hyperbolic_slots/data/slot_extractions_spot_coco")
OUT_PATH   = Path("/raid/create.nm43gr/hyperbolic_slots/data/gt_spot_coco.json")

SLOT_LEVELS = [3, 5, 7, 11, 13]
LEVEL_PAIRS = [(c, f) for c, f in combinations(SLOT_LEVELS, 2)]

# ── Read video dirs directly ──────────────────────────────────────────────────
vid_dirs = sorted(OUTPUT_DIR.glob("image_*/"))
print(f"Found {len(vid_dirs)} image dirs")
print(f"Level pairs: {LEVEL_PAIRS}\n")


def compute_inclusion_last_frame(masks_coarse, masks_fine, n_coarse, n_fine):
    coarse = masks_coarse[-1]
    fine   = masks_fine[-1]
    inclusion = torch.zeros(n_coarse, n_fine)
    for j in range(n_fine):
        mask_j = (fine == j)
        area_j = mask_j.sum().float()
        if area_j == 0:
            continue
        for i in range(n_coarse):
            overlap = (mask_j & (coarse == i)).sum().float()
            inclusion[i, j] = overlap / area_j
    return inclusion


def inclusion_matrix_to_tree(inclusion_matrix, n_coarse, n_fine):
    tree = {f"parent_{i}": [] for i in range(n_coarse)}
    for j in range(n_fine):
        best_i = inclusion_matrix[:, j].argmax().item()
        tree[f"parent_{best_i}"].append(f"child_{j}")
    return {k: v for k, v in tree.items() if v}


# ── Build GT dict ─────────────────────────────────────────────────────────────
gt_dict = {}

for vid_dir in tqdm(vid_dirs, desc="Computing GT trees"):

    # Use ytvis_key from meta as the dict key (matches other JSON files)
    vid_key = vid_dir.name 

    try:
        masks = {
            n: torch.load(vid_dir / f"masks_hard_{n}.pt", weights_only=True)
            for n in SLOT_LEVELS
        }

        vid_trees = {}
        for n_coarse, n_fine in LEVEL_PAIRS:
            inc = compute_inclusion_last_frame(
                masks[n_coarse], masks[n_fine], n_coarse, n_fine
            )
            vid_trees[f"{n_coarse}_to_{n_fine}"] = inclusion_matrix_to_tree(
                inc, n_coarse, n_fine
            )

        gt_dict[vid_key] = vid_trees

    except Exception as e:
        print(f"[SKIP] {vid_dir.name} (key={vid_key}): {e}")

# ── Verify first entry ────────────────────────────────────────────────────────
first_key = next(iter(gt_dict))
print(f"\nVerification — {first_key}:")
for pair_key, tree in gt_dict[first_key].items():
    n_coarse, n_fine = map(int, pair_key.split("_to_"))
    n_assigned = sum(len(v) for v in tree.values())
    print(f"  {pair_key:10s}  {n_assigned}/{n_fine} children assigned  |  {len(tree)} parents active")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {"gt_spot_coco": gt_dict}
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {OUT_PATH}  ({len(gt_dict)} images)")