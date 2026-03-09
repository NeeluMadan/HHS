"""
SPOT (SmoothSA) — Multi-Level Slot & Mask Extraction on COCO
=============================================================
Loads COCO val lmdb, runs SPOT at N = 3, 5, 7, 11, 13 slots,
and saves slots + masks per image.

Output layout:
  slot_extractions_spot_coco/
    image_000000/
      slots_raw_3.pt    (3,  256)   unnormalized
      slots_norm_3.pt   (3,  256)   L2-normalized
      masks_soft_3.pt   (3,  H, W)  slot attention (attenta)
      masks_hard_3.pt   (H,  W)     argmax label map
      slots_raw_5.pt    ...
      ...
      masks_hard_13.pt  (H,  W)
      meta.pt           {image_idx, n_slots_levels}
    image_000001/
      ...

Usage:
  python extract_slots_spot_coco.py
"""

import sys
import importlib
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm

# ── Edit these ────────────────────────────────────────────────────────────────
CHECKPOINT_PATH  = "./checkpoints/spot_coco.pth"
CONFIG_PATH      = "./baselines/SmoothSA/config-spot/spot_r-coco.py"
SMOOTHSA_ROOT    = "./baselines/SmoothSA"

COCO_LMDB        = "/root/datasets/AICF/lmdb/coco/val.lmdb"          # path to COCO val lmdb
OUTPUT_DIR       = Path("/raid/create.nm43gr/hyperbolic_slots/data/slot_extractions_spot_coco")

SLOT_LEVELS      = [3, 5, 7, 11, 13]
MAX_IMAGES       = None                        # set to int to cap
BATCH_SIZE       = 1                           # keep at 1 for simplicity
IMG_SIZE         = 256                         # must match resolut0 in config
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, SMOOTHSA_ROOT)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from object_centric_bench.util import Config, build_from_config
from object_centric_bench.model import ModelWrap

def load_config(config_path):
    return Config.fromfile(Path(config_path))

def load_model(checkpoint_path, config_path, device):
    cfg   = load_config(config_path)
    model = build_from_config(cfg.model)           # ← build_from_config, not build
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)
    model.load(checkpoint_path, cfg.ckpt_map)      # ← ModelWrap.load handles state dict
    model.eval().to(device)
    inner = model.m                                # ← actual SPOT is under .m
    print(f"Model loaded | default slots={inner.initializ.num} | dim={inner.initializ.dim}")
    return model

def patch_n_slots(model, n_slots):
    model.m.initializ.num = n_slots               # ← inner model is model.m


# ── Build COCO val dataset ────────────────────────────────────────────────────
def build_dataset(cfg, lmdb_path):
    dataset_cfg              = dict(cfg.dataset_v)
    dataset_cfg["data_file"] = lmdb_path
    dataset_cfg["base_dir"]  = Path(lmdb_path).parent
    return build_from_config(dataset_cfg)   # ← was build_from_cfg

# ── Extract all levels for one image ─────────────────────────────────────────
IMG_SIZE = 256   # must match resolut0 in config

@torch.no_grad()
def extract_all_levels(model, image_tensor, slot_levels, device):
    image_tensor = image_tensor.to(device)
    results = {}

    for n in slot_levels:
        patch_n_slots(model, n)

        output  = model(batch={"image": image_tensor})
        slotz   = output["slotz"]    # (1, N, D)
        attenta = output["attenta"]  # (1, N, 16, 16)

        slots_raw  = slotz[0].cpu()
        slots_norm = F.normalize(slots_raw, p=2, dim=-1)

        # upsample attention to full image resolution
        masks_soft = F.interpolate(
            attenta, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )[0].cpu()                                          # (N, 256, 256)
        masks_hard = masks_soft.argmax(dim=0)              # (256, 256)

        results[n] = {
            "slots_raw":  slots_raw,
            "slots_norm": slots_norm,
            "masks_soft": masks_soft,
            "masks_hard": masks_hard,
        }

    return results


def save_image(output_dir, image_idx, extractions, meta):
    img_dir = Path(output_dir) / f"image_{image_idx:06d}"
    img_dir.mkdir(parents=True, exist_ok=True)

    for n, ext in extractions.items():
        torch.save(ext["slots_raw"],  img_dir / f"slots_raw_{n}.pt")
        torch.save(ext["slots_norm"], img_dir / f"slots_norm_{n}.pt")
        torch.save(ext["masks_soft"], img_dir / f"masks_soft_{n}.pt")
        torch.save(ext["masks_hard"], img_dir / f"masks_hard_{n}.pt")

    torch.save(meta, img_dir / "meta.pt")


def main():
    print(f"Device      : {DEVICE}")
    print(f"Slot levels : {SLOT_LEVELS}")
    print(f"Output dir  : {OUTPUT_DIR}\n")

    model = load_model(CHECKPOINT_PATH, CONFIG_PATH, DEVICE)
    cfg   = load_config(CONFIG_PATH)

    # ── Sanity check patching ─────────────────────────────────────────────────
    print("Checking slot patching:")
    print(f"  Initializer type : {type(model.m.initializ).__name__}")
    for n in SLOT_LEVELS:
        patch_n_slots(model, n)
        print(f"  n={n:2d} | num={model.m.initializ.num}")
    print("\nPatch OK — building dataset\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = build_dataset(cfg, COCO_LMDB)
    print(f"COCO val images: {len(dataset)}\n")

    image_idx = 0

    for sample in tqdm(dataset, desc="Images"):

        if MAX_IMAGES is not None and image_idx >= MAX_IMAGES:
            break

        try:
            # image: (3, H, W) float32 after transform, already normalised
            image = sample["image"]
            if image.dtype == torch.uint8:
                image = image.float()
            image_tensor = image.unsqueeze(0)   # (1, 3, H, W)

            extractions = extract_all_levels(model, image_tensor, SLOT_LEVELS, DEVICE)

            meta = {
                "image_idx":   image_idx,
                "slot_levels": SLOT_LEVELS,
                "image_shape": tuple(image.shape),
            }

            save_image(OUTPUT_DIR, image_idx, extractions, meta)

            if image_idx % 200 == 0:
                shapes = {n: extractions[n]["masks_hard"].shape for n in SLOT_LEVELS}
                print(f"  [{image_idx}] masks_hard shapes: {shapes}")

        except Exception as e:
            import traceback
            print(f"[SKIP] image {image_idx}: {e}")
            traceback.print_exc()

        finally:
            image_idx += 1

    saved = len(sorted(OUTPUT_DIR.glob("image_*/meta.pt")))
    print(f"\nDone. {saved} images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()