"""
Gaussian Elimination with Partial Pivoting and Back-Substitution.
Solves the system Ax = b represented as an augmented matrix [A|b].
"""

import numpy as np
from typing import List

# Optional: Set a reasonable upper limit to prevent resource exhaustion.
# Matrices larger than this will be rejected. Adjust as needed.
MAX_MATRIX_SIZE = 100


def gaussian_elimination(matrix: List[List[float]]) -> dict:
    """
    Solve Ax = b using Gaussian elimination with partial pivoting
    and back-substitution.

    Args:
        matrix: Augmented matrix [A|b] of shape n × (n+1).

    Returns:
        dict with keys:
            solution   – list of n floats (x1 … xn)
            variables  – list of strings ['x1', 'x2', …]
            steps      – ordered list of step dicts
            n          – system size

    Raises:
        ValueError: If matrix is malformed, singular, or exceeds size limit.
    """
    n = len(matrix)
    if n == 0:
        raise ValueError("Matrix must have at least one row.")
    if n > MAX_MATRIX_SIZE:
        raise ValueError(
            f"Matrix size {n} exceeds maximum allowed size {MAX_MATRIX_SIZE}. "
            f"Use smaller systems to avoid resource exhaustion."
        )
    for i, row in enumerate(matrix):
        if len(row) != n + 1:
            raise ValueError(
                f"Row {i + 1} has {len(row)} elements; expected {n + 1} "
                f"(augmented matrix needs n+1 columns for n equations)."
            )

    aug = np.array(matrix, dtype=float)
    steps: list = []

    # ── Step 0: record the original augmented matrix ──────────────────────
    steps.append(
        {
            "type": "initial",
            "description": "Original augmented matrix  [A | b]",
            "matrix": _mat_to_list(aug),
        }
    )

    # ── Condition number check ─────────────────────────────────────────────
    try:
        cond_number = float(np.linalg.cond(aug[:, :n]))
    except Exception:
        cond_number = float("inf")
    if cond_number > 1e6:
        steps.append(
            {
                "type": "warning",
                "description": (
                    f"High condition number detected ({cond_number:.3g}). "
                    f"This matrix may be ill-conditioned — results could "
                    f"have reduced accuracy due to floating-point rounding."
                ),
            }
        )

    # ── Forward elimination with partial pivoting ──────────────────────────
    for col in range(n):
        # Find row with largest absolute value in current column (from col down)
        pivot_row = col + int(np.argmax(np.abs(aug[col:, col])))

        if abs(aug[pivot_row, col]) < 1e-12:
            raise ValueError(
                "Matrix is singular (or nearly singular) — "
                "the system has no unique solution."
            )

        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]
            steps.append(
                {
                    "type": "pivot",
                    "description": (
                        f"Partial pivot: swap R{col + 1} ↔ R{pivot_row + 1}  "
                        f"(largest pivot = {aug[col, col]:.6g})"
                    ),
                    "matrix": _mat_to_list(aug),
                    "pivot_row": col,
                    "pivot_col": col,
                }
            )

        pivot = aug[col, col]

        for row in range(col + 1, n):
            if abs(aug[row, col]) < 1e-15:
                continue
            factor = aug[row, col] / pivot
            aug[row] = aug[row] - factor * aug[col]

            steps.append(
                {
                    "type": "elimination",
                    "description": (
                        f"R{row + 1}  ←  R{row + 1}  −  "
                        f"({factor:.6g}) × R{col + 1}"
                    ),
                    "matrix": _mat_to_list(aug),
                    "pivot_row": col,
                    "pivot_col": col,
                    "active_row": row,
                }
            )

    steps.append(
        {
            "type": "upper_triangular",
            "description": "Upper-triangular form achieved — ready for back-substitution",
            "matrix": _mat_to_list(aug),
        }
    )

    # ── Back-substitution ──────────────────────────────────────────────────
    x = np.zeros(n)
    back_steps: list = []

    for i in range(n - 1, -1, -1):
        known_sum = float(np.dot(aug[i, i + 1 : n], x[i + 1 : n]))
        x[i] = (aug[i, n] - known_sum) / aug[i, i]

        # Build a readable formula string
        sub_parts = [
            f"({aug[i, j]:.6g})·({x[j]:.6g})"
            for j in range(i + 1, n)
            if abs(aug[i, j]) > 1e-15
        ]
        if sub_parts:
            formula = (
                f"( {aug[i, n]:.6g} − {' − '.join(sub_parts)} ) "
                f"/ {aug[i, i]:.6g}"
            )
        else:
            formula = f"{aug[i, n]:.6g} / {aug[i, i]:.6g}"

        back_steps.append(
            {
                "variable": f"x{i + 1}",
                "value": _round(x[i]),
                "description": f"x{i + 1} = {formula} = {x[i]:.8g}",
            }
        )

    # Reverse so back-steps are shown x1 → xn in the UI
    back_steps.reverse()

    steps.append(
        {
            "type": "back_substitution",
            "description": "Back-substitution — solving for each variable",
            "back_steps": back_steps,
        }
    )

    return {
        "solution": [_round(xi) for xi in x],
        "variables": [f"x{i + 1}" for i in range(n)],
        "steps": steps,
        "n": n,
    }


# ── helpers ────────────────────────────────────────────────────────────────

def _round(v: float, decimals: int = 8) -> float:
    r = round(float(v), decimals)
    # Avoid -0.0 in output
    return 0.0 if r == 0.0 else r


def _mat_to_list(arr: np.ndarray) -> list:
    """Convert numpy 2-D array to a list of lists of rounded floats."""
    return [[_round(v, 6) for v in row] for row in arr.tolist()]
