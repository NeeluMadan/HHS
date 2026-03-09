"""Compute parent-child trees and distance matrices for videosaur-dino-v1 slots.

For each video, takes the last frame's slot embeddings and computes trees
on multiple manifolds.

Outputs:
  - JSON file with parent-child tree assignments
  - .pt file with full (1+P+C)x(1+P+C) upper-triangle pairwise distance matrices
    per manifold, plus n_parents and n_children metadata.
    Row/col 0 = origin, 1..P = parents, P+1..P+C = children.
"""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from hyperbolic_tree import (
    MANIFOLD_REGISTRY,
    build_parent_child_trees,
    dist_from_origin,
    pairwise_distances,
    project_to_manifold,
)

import torch

MANIFOLDS = [
    "sphere",
    "lorentz_1_exp",
    "lorentz_1_projx",
    "euclidean",
    # --- commented out for easy swap between experiments ---
    # "lorentz_01",
    # "lorentz_05",
    # "lorentz_1",
    # "lorentz_2",
    # "lorentz_5",
    # "poincare_01",
    # "poincare_05",
    # "poincare_1",
    # "poincare_2",
    # "poincare_5",
]


def load_last_frame(video_dir: Path):
    """Load parent and child slot embeddings for the last frame."""
    parents = torch.load(video_dir / "slots_raw_7.pt", weights_only=True)
    children = torch.load(video_dir / "slots_raw_15.pt", weights_only=True)
    return parents[-1].double(), children[-1].double()


def compute_trees_and_dists(parent_vecs, child_vecs):
    """Compute trees and full pairwise distance matrices for all manifolds.

    Returns:
        trees: dict mapping manifold name -> {parent_label: set(child_labels)}
        dists: dict with "n_parents", "n_children", and per-manifold
               (1+P+C, 1+P+C) upper-triangle distance matrices.
               Row/col 0 = origin, 1..P = parents, P+1..P+C = children.
    """
    P = parent_vecs.shape[0]
    C = child_vecs.shape[0]
    parent_labels = [f"parent_{i}" for i in range(P)]
    child_labels = [f"child_{j}" for j in range(C)]

    trees = {}
    dists = {"n_parents": P, "n_children": C}

    for name in MANIFOLDS:
        manifold = MANIFOLD_REGISTRY[name]
        manifold = manifold.to(dtype=parent_vecs.dtype, device=parent_vecs.device)

        method = "projx" if name.endswith("_projx") else "expmap0"
        parents_proj = project_to_manifold(parent_vecs, manifold, method=method)
        children_proj = project_to_manifold(child_vecs, manifold, method=method)

        # Full (P+C) x (P+C) pairwise distance matrix
        all_proj = torch.cat([parents_proj, children_proj], dim=0)
        full_dist = pairwise_distances(all_proj, all_proj, manifold)

        # Distance from origin for all points
        origin_dists = dist_from_origin(all_proj, manifold)  # (P+C,)

        # Build (1+P+C) x (1+P+C) matrix: row/col 0 = origin
        N = P + C
        full_with_origin = torch.zeros(1 + N, 1 + N, dtype=full_dist.dtype)
        full_with_origin[0, 1:] = origin_dists
        full_with_origin[1:, 0] = origin_dists
        full_with_origin[1:, 1:] = full_dist
        dists[name] = torch.triu(full_with_origin)

        # Tree building uses the parent-child submatrix
        dist_pc = full_dist[:P, P:]
        tree = build_parent_child_trees(
            parents_proj, children_proj, manifold,
            parent_labels=parent_labels,
            child_labels=child_labels,
        )
        trees[name] = tree

    return trees, dists


def compute_agreement(trees_all_videos):
    """Compute pairwise agreement between manifolds across all videos.

    Agreement = fraction of children assigned to the same parent in both manifolds.
    """
    pair_agree = defaultdict(list)

    for video_name, manifold_trees in trees_all_videos.items():
        # Build child -> parent assignment for each manifold
        assignments = {}
        for mname, tree in manifold_trees.items():
            child_to_parent = {}
            for parent, children in tree.items():
                for child in children:
                    child_to_parent[child] = parent
            assignments[mname] = child_to_parent

        # Compare each pair of manifolds
        for m1, m2 in combinations(MANIFOLDS, 2):
            a1, a2 = assignments[m1], assignments[m2]
            all_children = set(a1.keys()) | set(a2.keys())
            if not all_children:
                continue
            agree = sum(1 for c in all_children if a1.get(c) == a2.get(c))
            pair_agree[(m1, m2)].append(agree / len(all_children))

    return {
        f"{m1}_vs_{m2}": sum(vals) / len(vals)
        for (m1, m2), vals in pair_agree.items()
    }


def print_summary(trees_all_videos, agreement):
    """Print per-manifold statistics and cross-manifold agreement."""
    n_videos = len(trees_all_videos)
    print(f"\n{'='*60}")
    print(f"Summary over {n_videos} videos")
    print(f"{'='*60}")

    # Per-manifold: average children per parent
    for mname in MANIFOLDS:
        counts = []
        for video_trees in trees_all_videos.values():
            tree = video_trees[mname]
            counts.extend(len(children) for children in tree.values())
        avg = sum(counts) / len(counts) if counts else 0
        std = (sum((c - avg) ** 2 for c in counts) / len(counts)) ** 0.5 if counts else 0
        print(f"\n  {mname:10s}: avg children/parent = {avg:.2f} +/- {std:.2f}")

    # Cross-manifold agreement
    print(f"\n  Cross-manifold agreement (fraction of children assigned to same parent):")
    for pair_name, avg_agree in sorted(agreement.items()):
        print(f"    {pair_name:25s}: {avg_agree:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute parent-child trees for videosaur-dino-v1 slots."
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="/home/alexpv/Default_Folder/notes/phd/1-projects/videosaur-dino-v1",
        help="Path to videosaur-dino-v1 data directory.",
    )
    parser.add_argument(
        "--output-trees", type=str, default=None,
        help="Path to save trees JSON (default: <script_dir>/videosaur_dino_v1_trees.json).",
    )
    parser.add_argument(
        "--output-dists", type=str, default=None,
        help="Path to save distance matrices .pt (default: <script_dir>/videosaur_dino_v1_dists.pt).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    script_dir = Path(__file__).parent
    output_trees = Path(args.output_trees) if args.output_trees else script_dir / "videosaur_dino_v1_trees.json"
    output_dists = Path(args.output_dists) if args.output_dists else script_dir / "videosaur_dino_v1_dists.pt"

    video_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("video_"))
    print(f"Found {len(video_dirs)} videos in {data_dir}")

    trees_all = {}
    dists_all = {}

    for i, vdir in enumerate(video_dirs):
        parent_vecs, child_vecs = load_last_frame(vdir)
        trees, dists = compute_trees_and_dists(parent_vecs, child_vecs)

        video_name = vdir.name
        # Convert sets to sorted lists for JSON
        trees_all[video_name] = {
            mname: {p: sorted(c) for p, c in tree.items()}
            for mname, tree in trees.items()
        }
        dists_all[video_name] = dists

        if (i + 1) % 50 == 0 or (i + 1) == len(video_dirs):
            print(f"  Processed {i + 1}/{len(video_dirs)} videos")

    # Save trees as JSON
    with open(output_trees, "w") as f:
        json.dump(trees_all, f, indent=2)
    print(f"\nTrees saved to {output_trees}")

    # Save distance matrices as .pt
    torch.save(dists_all, output_dists)
    print(f"Distance matrices saved to {output_dists}")

    # Compute and print summary
    # Re-parse trees_all into set form for agreement computation
    trees_sets = {}
    for vname, mtrees in trees_all.items():
        trees_sets[vname] = {
            mname: {p: set(children) for p, children in tree.items()}
            for mname, tree in mtrees.items()
        }

    agreement = compute_agreement(trees_sets)
    print_summary(trees_sets, agreement)


if __name__ == "__main__":
    main()
