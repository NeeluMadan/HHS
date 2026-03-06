import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

def compute_agreement(trees_all_videos, manifolds):
    """Compute pairwise agreement between manifolds across all videos.

    Agreement = fraction of children assigned to the same parent in both manifolds.
    """
    pair_agree = defaultdict(list)

    for video_name, manifold_trees in trees_all_videos.items():
        assignments = {}
        for mname, tree in manifold_trees.items():
            child_to_parent = {}
            for parent, children in tree.items():
                for child in children:
                    child_to_parent[child] = parent
            assignments[mname] = child_to_parent

        for m1, m2 in combinations(manifolds, 2):
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


def print_summary(trees_all_videos, agreement, manifolds):
    """Print per-manifold statistics and cross-manifold agreement."""
    n_videos = len(trees_all_videos)
    print(f"\n{'='*60}")
    print(f"Summary over {n_videos} videos")
    print(f"{'='*60}")

    for mname in manifolds:
        counts = []
        for video_trees in trees_all_videos.values():
            tree = video_trees[mname]
            counts.extend(len(children) for children in tree.values())
        avg = sum(counts) / len(counts) if counts else 0
        std = (sum((c - avg) ** 2 for c in counts) / len(counts)) ** 0.5 if counts else 0
        print(f"\n  {mname:20s}: avg children/parent = {avg:.2f} +/- {std:.2f}")

    print(f"\n  Cross-manifold agreement (fraction of children assigned to same parent):")
    for pair_name, avg_agree in sorted(agreement.items()):
        print(f"    {pair_name:40s}: {avg_agree:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Comparison between parent-child trees for videosaur-dino-v1 slots."
    )
    parser.add_argument(
        "--trees", type=str, required=True,
        help="Path to trees JSON.",
    )
    parser.add_argument(
        "--gt-trees", type=str, default=None,
        help="Path to ground-truth trees JSON (default: <script_dir>/videosaur_dino_v1_gt_trees.json).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    trees_path = Path(args.trees)
    gt_path = Path(args.gt_trees) if args.gt_trees else script_dir / "videosaur_dino_v1_gt_trees.json"

    with open(trees_path, "r") as f:
        trees_all = json.load(f)
    print(f"\nTrees loaded from {trees_path}")

    with open(gt_path, "r") as f:
        gt_trees = json.load(f)
    print(f"GT loaded from {gt_path}")

    # Merge GT into each video
    for vname in trees_all:
        if vname in gt_trees:
            trees_all[vname]["gt"] = gt_trees[vname]

    # Convert lists to sets for agreement computation
    trees_sets = {}
    for vname, mtrees in trees_all.items():
        trees_sets[vname] = {
            mname: {p: set(children) for p, children in tree.items()}
            for mname, tree in mtrees.items()
        }

    manifolds = list(next(iter(trees_sets.values())).keys())
    agreement = compute_agreement(trees_sets, manifolds)
    print_summary(trees_sets, agreement, manifolds)


if __name__ == "__main__":
    main()
