"""Project vectors into the Lorentz (hyperbolic) manifold and build parent-child trees."""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add local geoopt checkout to the import path.
_GEOOPT_ROOT = str(
    Path("/home/alexpv/Default_Folder/notes/phd/1-projects")
    / "video-language-representation-learning"
    / "geoopt"
)
sys.path.insert(0, _GEOOPT_ROOT)

import torch
import torch.nn.functional as F
from geoopt.manifolds.lorentz import Lorentz


def project_to_manifold(vectors: torch.Tensor, manifold: Lorentz) -> torch.Tensor:
    """Map vectors from tangent space at the origin onto the Lorentz manifold.

    Prepends a zeros column (time dimension) and applies the exponential map
    from the origin (expmap0), which is the geometrically correct way to map
    Euclidean vectors onto the hyperboloid.

    Args:
        vectors: Tensor of shape (N, D) in Euclidean space.
        manifold: A geoopt Lorentz manifold instance.

    Returns:
        Tensor of shape (N, D+1) on the hyperboloid.
    """
    if vectors.shape[0] == 0:
        return vectors.new_empty(0, vectors.shape[-1] + 1)
    padded = F.pad(vectors, (1, 0, 0, 0))  # (N, D) -> (N, D+1) with zero at index 0
    return manifold.expmap0(padded)


def build_parent_child_trees(
    parents: torch.Tensor,
    children: torch.Tensor,
    manifold: Lorentz,
    parent_labels: Optional[List[str]] = None,
    child_labels: Optional[List[str]] = None,
) -> Dict[str, Set[str]]:
    """Build parent-child trees by assigning each child to its nearest parent.

    Uses manifold.cdist to compute pairwise geodesic distances, then assigns
    each child to the parent with smallest distance.

    Args:
        parents: Tensor of shape (P, D) on the Lorentz manifold.
        children: Tensor of shape (C, D) on the Lorentz manifold.
        manifold: A geoopt Lorentz manifold instance.
        parent_labels: Optional list of P string labels for parents.
        child_labels: Optional list of C string labels for children.

    Returns:
        Dict mapping parent labels to sets of child labels.
    """
    P = parents.shape[0]
    C = children.shape[0]

    if parent_labels is None:
        parent_labels = [f"parent_{i}" for i in range(P)]
    if child_labels is None:
        child_labels = [f"child_{j}" for j in range(C)]

    assert parents.shape[-1] == children.shape[-1], (
        f"Dimension mismatch: parents have {parents.shape[-1]}, "
        f"children have {children.shape[-1]}."
    )

    tree: Dict[str, Set[str]] = {label: set() for label in parent_labels}

    if P == 0 or C == 0:
        return tree

    # Pairwise geodesic distances: shape (P, C)
    dist_matrix = manifold.cdist(parents, children)

    # For each child, find the nearest parent
    nearest_parent_indices = torch.argmin(dist_matrix, dim=0)  # shape (C,)

    for j, parent_idx in enumerate(nearest_parent_indices.tolist()):
        tree[parent_labels[parent_idx]].add(child_labels[j])

    return tree


def build_trees(
    parent_vectors: torch.Tensor,
    child_vectors: torch.Tensor,
    curvature: float = 1.0,
    parent_labels: Optional[List[str]] = None,
    child_labels: Optional[List[str]] = None,
) -> Dict[str, Set[str]]:
    """End-to-end pipeline: project vectors and build parent-child trees.

    Args:
        parent_vectors: Tensor of shape (P, D) in Euclidean space.
        child_vectors: Tensor of shape (C, D) in Euclidean space.
        curvature: Negative curvature k for the Lorentz manifold (default: 1.0).
        parent_labels: Optional string labels for parents.
        child_labels: Optional string labels for children.

    Returns:
        Dict mapping parent labels to sets of assigned child labels.
    """
    if parent_vectors.dtype != torch.float64:
        warnings.warn(
            "Lorentz manifold operations are more stable in float64. "
            "Consider passing .double() tensors.",
            stacklevel=2,
        )

    manifold = Lorentz(k=curvature, learnable=False)
    manifold = manifold.to(dtype=parent_vectors.dtype, device=parent_vectors.device)

    parents_proj = project_to_manifold(parent_vectors, manifold)
    children_proj = project_to_manifold(child_vectors, manifold)

    return build_parent_child_trees(
        parents_proj, children_proj, manifold,
        parent_labels=parent_labels,
        child_labels=child_labels,
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Build parent-child trees in Lorentz (hyperbolic) space."
    )
    parser.add_argument(
        "--parents", type=str, required=True,
        help="Path to a .pt file containing parent vectors (P, D).",
    )
    parser.add_argument(
        "--children", type=str, required=True,
        help="Path to a .pt file containing child vectors (C, D).",
    )
    parser.add_argument(
        "--curvature", type=float, default=1.0,
        help="Negative curvature k of the Lorentz manifold (default: 1.0).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the output tree as a JSON file.",
    )
    args = parser.parse_args()

    parent_vecs = torch.load(args.parents, weights_only=True)
    child_vecs = torch.load(args.children, weights_only=True)

    trees = build_trees(parent_vecs, child_vecs, curvature=args.curvature)

    # Convert sets to sorted lists for JSON serialization
    trees_json = {k: sorted(v) for k, v in trees.items()}
    output_str = json.dumps(trees_json, indent=2)
    print(output_str)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        print(f"\nTree saved to {args.output}")
