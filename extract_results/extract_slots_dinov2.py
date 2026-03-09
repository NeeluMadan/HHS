"""
VideoSAUR DINOv2 — Multi-Level Slot & Mask Extraction Script
=============================================================
Reads YTVIS validation shards directly, resizes to 518×518,
and runs the model at N = 3, 5, 7, 11, 13 slots per video.

Output layout:
  slot_extractions_dinov2/
    video_0000/
      slots_raw_3.pt    (T,  3, 64)
      slots_norm_3.pt   (T,  3, 64)
      masks_soft_3.pt   (T,  3, P)
      masks_hard_3.pt   (T,  H, W)
      slots_raw_5.pt    ...
      ...
      masks_hard_13.pt  (T, 13, H, W)
      meta.pt
    video_0001/
      ...
"""

import sys
import torch
import torch.nn.functional as F
import webdataset as wds
from pathlib import Path
from tqdm.auto import tqdm
from torchvision import transforms as tvt

# ── Edit these ────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "./checkpoints/slotcontrast_ytvis.ckpt"
CONFIG_PATH     = "./baselines/slotcontrast/configs/slotcontrast/ytvis2021.yaml"
VIDEOSAUR_ROOT  = "./baselines/slotcontrast"

VAL_SHARDS  = "/root/datasets/AICF/ytvis2021_resized/ytvis-validation-{000000..000029}.tar"
OUTPUT_DIR  = Path("./slot_extractions_slotconstrast_dinov2")

SLOT_LEVELS = [3, 5, 7, 11, 13]   # ← all levels to extract
INPUT_SIZE  = 518
MAX_VIDEOS  = 300                  # set to None for all
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, VIDEOSAUR_ROOT)
from slotcontrast import configuration, models

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NORM   = tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
RESIZE = tvt.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True)


def load_model(checkpoint_path, config_path, device):
    config = configuration.load_config(config_path)
    model  = models.build(config.model, config.optimizer)
    ckpt   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # ← add this
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    print(f"Model loaded | default slots={model.initializer.n_slots} | dim={model.initializer.dim}")
    return model

def preprocess_video(video_np):
    video  = torch.from_numpy(video_np).float() / 255.0
    video  = video.permute(0, 3, 1, 2)                        # (T, 3, H, W)
    frames = [NORM(RESIZE(video[t])) for t in range(len(video))]
    return torch.stack(frames).unsqueeze(0)                   # (1, T, 3, 518, 518)

@torch.no_grad()
def extract_slots(model, video_tensor, n_slots, device):
    init = model.initializer

    # ── Patch both the count AND the parameter tensor ─────────────────────────
    init.n_slots = n_slots
    if hasattr(init, "slots") and isinstance(init.slots, torch.nn.Parameter):
        dim = init.slots.shape[-1]
        init.slots = torch.nn.Parameter(
            torch.empty(1, n_slots, dim, device=next(model.parameters()).device).normal_()
        )

    inputs      = {"video": video_tensor.to(device)}
    outputs     = model(inputs)
    aux_outputs = model.aux_forward(inputs, outputs)

    slots_raw  = outputs["processor"]["state"][0].cpu()
    slots_norm = F.normalize(slots_raw, p=2, dim=-1)

    soft = outputs["processor"]["corrector"].get("masks")
    if soft is not None:
        masks_soft = soft[0].cpu()
    else:
        dec = outputs["decoder"].get("masks")
        masks_soft = dec[0].cpu() if dec is not None else None

    hard_onehot = aux_outputs.get("decoder_masks_hard")
    if hard_onehot is not None:
        masks_hard = hard_onehot[0].cpu().argmax(dim=1)
    else:
        masks_hard = None

    return {"slots_raw": slots_raw, "slots_norm": slots_norm,
            "masks_soft": masks_soft, "masks_hard": masks_hard}

def save_video(output_dir, video_idx, extractions, meta):
    vid_dir = Path(output_dir) / f"video_{video_idx:04d}"
    vid_dir.mkdir(parents=True, exist_ok=True)

    for n, ext in extractions.items():
        torch.save(ext["slots_raw"],  vid_dir / f"slots_raw_{n}.pt")
        torch.save(ext["slots_norm"], vid_dir / f"slots_norm_{n}.pt")
        if ext["masks_soft"] is not None:
            torch.save(ext["masks_soft"], vid_dir / f"masks_soft_{n}.pt")
        if ext["masks_hard"] is not None:
            torch.save(ext["masks_hard"], vid_dir / f"masks_hard_{n}.pt")

    torch.save(meta, vid_dir / "meta.pt")


def main():
    print(f"Device      : {DEVICE}")
    print(f"Input size  : {INPUT_SIZE}×{INPUT_SIZE}")
    print(f"Slot levels : {SLOT_LEVELS}")
    print(f"Output dir  : {OUTPUT_DIR}\n")

    model = load_model(CHECKPOINT_PATH, CONFIG_PATH, DEVICE)
    
    # ── Sanity check: verify slot patching works before processing any video ──
    print("Checking slot initializer patching:")
    for n in SLOT_LEVELS:
        init = model.initializer
        init.n_slots = n
        if hasattr(init, "slots") and isinstance(init.slots, torch.nn.Parameter):
            dim = init.slots.shape[-1]
            init.slots = torch.nn.Parameter(
                torch.empty(1, n, dim, device=next(model.parameters()).device).normal_()
            )
        print(f"  n={n:2d} | n_slots={init.n_slots} | slots.shape={init.slots.shape}")
    print()
    # ─────────────────────────────────────────────────────────────────────────

    dataset = (
        wds.WebDataset(VAL_SHARDS, shardshuffle=False)
        .decode()
        .to_tuple("__key__", "video.npy", "segmentations.npy")
    )

    video_idx = 0

    for vid_key, video_np, seg_np in tqdm(dataset, desc="Videos"):

        if MAX_VIDEOS is not None and video_idx >= MAX_VIDEOS:
            break

        try:
            T            = video_np.shape[0]
            video_tensor = preprocess_video(video_np)

            # ── Run all levels on the same preprocessed tensor ────────────────
            extractions = {
                n: extract_slots(model, video_tensor, n_slots=n, device=DEVICE)
                for n in SLOT_LEVELS
            }

            meta = {
                "video_id":    video_idx,
                "ytvis_key":   vid_key,
                "n_frames":    T,
                "orig_shape":  video_np.shape,
                "slot_levels": SLOT_LEVELS,
            }

            save_video(OUTPUT_DIR, video_idx, extractions, meta)

            if video_idx % 20 == 0:
                shapes = {n: extractions[n]["masks_hard"].shape
                          for n in SLOT_LEVELS
                          if extractions[n]["masks_hard"] is not None}
                print(f"  [{video_idx}] key={vid_key} | T={T} | {shapes}")

        except Exception as e:
            import traceback
            print(f"[SKIP] video {video_idx} (key={vid_key}): {e}")
            traceback.print_exc()

        finally:
            video_idx += 1

    saved = len(sorted(OUTPUT_DIR.glob("video_*/meta.pt")))
    print(f"\nDone. {saved} videos saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()