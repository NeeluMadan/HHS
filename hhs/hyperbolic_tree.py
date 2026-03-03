"""Project vectors onto a Riemannian manifold and build parent-child trees.

Supported manifolds (via geoopt):
  - Lorentz (hyperboloid model)
  - Poincaré ball
  - Sphere
  - Euclidean
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

# Add local geoopt checkout to the import path.
_GEOOPT_ROOT = str(
    Path("/home/alexpv/Default_Folder/notes/phd/1-projects")
    / "video-language-representation-learning"
    / "geoopt"
)
sys.path.insert(0, _GEOOPT_ROOT)

import torch
import torch.nn.functional as F
from geoopt.manifolds.base import Manifold
from geoopt.manifolds.euclidean import Euclidean
from geoopt.manifolds.lorentz import Lorentz
from geoopt.manifolds.sphere import Sphere
from geoopt.manifolds.stereographic import PoincareBall


def project_to_manifold(
    vectors: torch.Tensor, manifold: Manifold, method: str = "expmap0"
) -> torch.Tensor:
    """Map Euclidean vectors onto the given manifold.

    Projection strategy depends on the manifold type and method:
      - Lorentz + expmap0: prepend zeros column + expmap0  (N, D) -> (N, D+1)
      - Lorentz + projx:   prepend zeros column + projx    (N, D) -> (N, D+1)
      - Poincaré ball: expmap0                             (N, D) -> (N, D)
      - Sphere: projx (normalize to unit norm)             (N, D) -> (N, D)
      - Euclidean: identity                                (N, D) -> (N, D)

    Args:
        vectors: Tensor of shape (N, D) in Euclidean space.
        manifold: A geoopt manifold instance.
        method: Projection method for Lorentz — "expmap0" or "projx".

    Returns:
        Tensor of shape (N, D') on the manifold.
    """
    if vectors.shape[0] == 0:
        extra = 1 if isinstance(manifold, Lorentz) else 0
        return vectors.new_empty(0, vectors.shape[-1] + extra)

    if isinstance(manifold, Lorentz):
        padded = F.pad(vectors, (1, 0, 0, 0))  # (N, D) -> (N, D+1)
        if method == "projx":
            return manifold.projx(padded)
        return manifold.expmap0(padded)
    elif isinstance(manifold, PoincareBall):
        return manifold.expmap0(vectors)
    elif isinstance(manifold, Sphere):
        return manifold.projx(vectors)
    elif isinstance(manifold, Euclidean):
        return vectors
    else:
        raise ValueError(f"Unsupported manifold type: {type(manifold).__name__}")


def pairwise_distances(
    x: torch.Tensor, y: torch.Tensor, manifold: Manifold
) -> torch.Tensor:
    """Compute pairwise geodesic distances between two sets of points.

    Uses manifold.cdist when available, otherwise falls back to broadcasting
    manifold.dist.

    Args:
        x: Tensor of shape (P, D) on the manifold.
        y: Tensor of shape (C, D) on the manifold.
        manifold: A geoopt manifold instance.

    Returns:
        Tensor of shape (P, C) of pairwise distances.
    """
    try:
        return manifold.cdist(x, y)
    except (NotImplementedError, TypeError):
        # Fallback: broadcast dist (e.g. PoincareBall has no cdist)
        return manifold.dist(x.unsqueeze(1), y.unsqueeze(0)).squeeze(-1)


def dist_from_origin(points: torch.Tensor, manifold: Manifold) -> torch.Tensor:
    """Compute geodesic distance from each point to the manifold origin.

    For Sphere, uses the normalized centroid of the input points as origin.

    Args:
        points: Tensor of shape (N, D) on the manifold.
        manifold: A geoopt manifold instance.

    Returns:
        Tensor of shape (N,) with distances to origin.
    """
    if isinstance(manifold, Lorentz):
        return manifold.dist0(points)
    elif isinstance(manifold, PoincareBall):
        return manifold.dist0(points)
    elif isinstance(manifold, Euclidean):
        return points.norm(dim=-1)
    elif isinstance(manifold, Sphere):
        mean = points.mean(dim=0)
        mean = mean / mean.norm()
        return manifold.dist(mean.unsqueeze(0), points).squeeze(-1)
    else:
        raise ValueError(f"Unsupported manifold type: {type(manifold).__name__}")


def build_parent_child_trees(
    parents: torch.Tensor,
    children: torch.Tensor,
    manifold: Manifold,
    parent_labels: Optional[List[str]] = None,
    child_labels: Optional[List[str]] = None,
) -> Dict[str, Set[str]]:
    """Build parent-child trees by assigning each child to its nearest parent.

    Computes pairwise geodesic distances, then assigns each child to the
    parent with the smallest distance.

    Args:
        parents: Tensor of shape (P, D) on the manifold.
        children: Tensor of shape (C, D) on the manifold.
        manifold: A geoopt manifold instance.
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
    dist_matrix = pairwise_distances(parents, children, manifold)

    # For each child, find the nearest parent
    nearest_parent_indices = torch.argmin(dist_matrix, dim=0)  # shape (C,)

    for j, parent_idx in enumerate(nearest_parent_indices.tolist()):
        tree[parent_labels[parent_idx]].add(child_labels[j])

    return tree


MANIFOLD_REGISTRY = {
    "sphere": Sphere(),
    "lorentz_1_exp": Lorentz(k=1.0, learnable=False),
    "lorentz_1_projx": Lorentz(k=1.0, learnable=False),
    "euclidean": Euclidean(ndim=1),
    # --- commented out for easy swap between experiments ---
    # "lorentz_01": Lorentz(k=0.1, learnable=False),
    # "lorentz_05": Lorentz(k=0.5, learnable=False),
    # "lorentz_1": Lorentz(k=1.0, learnable=False),
    # "lorentz_2": Lorentz(k=2.0, learnable=False),
    # "lorentz_5": Lorentz(k=5.0, learnable=False),
    # "poincare_01": PoincareBall(c=0.1, learnable=False),
    # "poincare_05": PoincareBall(c=0.5, learnable=False),
    # "poincare_1": PoincareBall(c=1.0, learnable=False),
    # "poincare_2": PoincareBall(c=2.0, learnable=False),
    # "poincare_5": PoincareBall(c=5.0, learnable=False),
}


def build_trees(
    parent_vectors: torch.Tensor,
    child_vectors: torch.Tensor,
    manifold: Union[str, Manifold] = "lorentz",
    curvature: float = 1.0,
    parent_labels: Optional[List[str]] = None,
    child_labels: Optional[List[str]] = None,
) -> Dict[str, Set[str]]:
    """End-to-end pipeline: project vectors and build parent-child trees.

    Args:
        parent_vectors: Tensor of shape (P, D) in Euclidean space.
        child_vectors: Tensor of shape (C, D) in Euclidean space.
        manifold: Manifold name (str) or a geoopt Manifold instance.
            Supported names: "lorentz", "poincare", "sphere", "euclidean".
        curvature: Curvature parameter (only used when manifold is a string;
            maps to k for Lorentz, c for Poincaré; ignored for Sphere/Euclidean).
        parent_labels: Optional string labels for parents.
        child_labels: Optional string labels for children.

    Returns:
        Dict mapping parent labels to sets of assigned child labels.
    """
    # Construct manifold if given as string
    if isinstance(manifold, str):
        name = manifold.lower()
        if name not in MANIFOLD_REGISTRY:
            raise ValueError(
                f"Unknown manifold '{manifold}'. "
                f"Choose from: {list(MANIFOLD_REGISTRY.keys())}"
            )
        manifold = MANIFOLD_REGISTRY[name]

    if isinstance(manifold, (Lorentz, PoincareBall)) and parent_vectors.dtype != torch.float64:
        warnings.warn(
            f"{type(manifold).__name__} manifold operations are more stable in float64. "
            "Consider passing .double() tensors.",
            stacklevel=2,
        )

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
        description="Build parent-child trees on a Riemannian manifold."
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
        "--manifold", type=str, default="lorentz",
        choices=list(MANIFOLD_REGISTRY.keys()),
        help="Manifold type (default: lorentz).",
    )
    parser.add_argument(
        "--curvature", type=float, default=1.0,
        help="Curvature parameter (k for Lorentz, c for Poincaré; default: 1.0).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the output tree as a JSON file.",
    )
    args = parser.parse_args()

    parent_vecs = torch.load(args.parents, weights_only=True)
    child_vecs = torch.load(args.children, weights_only=True)

    trees = build_trees(
        parent_vecs, child_vecs,
        manifold=args.manifold, curvature=args.curvature,
    )

    # Convert sets to sorted lists for JSON serialization
    trees_json = {k: sorted(v) for k, v in trees.items()}
    output_str = json.dumps(trees_json, indent=2)
    print(output_str)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        print(f"\nTree saved to {args.output}")
