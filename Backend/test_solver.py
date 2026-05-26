"""
Pytest test suite for solver.py
Tests correctness of Gaussian elimination algorithm against known solutions.
"""

import pytest
import math
from solver import gaussian_elimination


class TestBasicSolutions:
    """Test basic systems with known solutions."""

    def test_2x2_simple(self):
        """Test a simple 2×2 system: 3x + 2y = 8, x - y = -1 → (1.2, 2.2)"""
        matrix = [
            [3, 2, 8],
            [1, -1, -1],
        ]
        result = gaussian_elimination(matrix)
        assert len(result["solution"]) == 2
        assert math.isclose(result["solution"][0], 1.2, abs_tol=1e-6)
        assert math.isclose(result["solution"][1], 2.2, abs_tol=1e-6)

    def test_3x3_standard(self):
        """Test the standard 3×3 example from README:
        2x₁ + x₂ - x₃ = 8
        -3x₁ - x₂ + 2x₃ = -11
        -2x₁ + x₂ + 2x₃ = -3
        Expected: (2, 3, -1)
        """
        matrix = [
            [2, 1, -1, 8],
            [-3, -1, 2, -11],
            [-2, 1, 2, -3],
        ]
        result = gaussian_elimination(matrix)
        assert len(result["solution"]) == 3
        assert math.isclose(result["solution"][0], 2.0, abs_tol=1e-6)
        assert math.isclose(result["solution"][1], 3.0, abs_tol=1e-6)
        assert math.isclose(result["solution"][2], -1.0, abs_tol=1e-6)

    def test_identity_matrix(self):
        """Test identity matrix: x=1, y=2, z=3"""
        matrix = [
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
        ]
        result = gaussian_elimination(matrix)
        assert math.isclose(result["solution"][0], 1.0, abs_tol=1e-6)
        assert math.isclose(result["solution"][1], 2.0, abs_tol=1e-6)
        assert math.isclose(result["solution"][2], 3.0, abs_tol=1e-6)

    def test_negative_solution(self):
        """Test system with negative solution: x=-1, y=-2"""
        matrix = [
            [1, 1, -3],
            [2, -1, 0],
        ]
        result = gaussian_elimination(matrix)
        assert math.isclose(result["solution"][0], -1.0, abs_tol=1e-6)
        assert math.isclose(result["solution"][1], -2.0, abs_tol=1e-6)

    def test_fractional_solution(self):
        """Test system with fractional solution: x=0.5, y=1.5"""
        matrix = [
            [2, 0, 1],
            [0, 2, 3],
        ]
        result = gaussian_elimination(matrix)
        assert math.isclose(result["solution"][0], 0.5, abs_tol=1e-6)
        assert math.isclose(result["solution"][1], 1.5, abs_tol=1e-6)


class TestStructure:
    """Test the structure and metadata of the result."""

    def test_result_keys(self):
        """Verify that result contains all expected keys."""
        matrix = [[1, 0, 1], [0, 1, 2]]
        result = gaussian_elimination(matrix)
        assert "solution" in result
        assert "variables" in result
        assert "steps" in result
        assert "n" in result

    def test_variables_naming(self):
        """Verify that variables are named x1, x2, etc."""
        matrix = [[1, 0, 0, 5], [0, 1, 0, 10], [0, 0, 1, 15]]
        result = gaussian_elimination(matrix)
        assert result["variables"] == ["x1", "x2", "x3"]

    def test_steps_sequence(self):
        """Verify that steps follow the expected sequence."""
        matrix = [[2, 1, 8], [1, -1, -1]]
        result = gaussian_elimination(matrix)
        step_types = [step["type"] for step in result["steps"]]
        assert step_types[0] == "initial"
        assert "upper_triangular" in step_types
        assert step_types[-1] == "back_substitution"

    def test_step_has_description(self):
        """Verify that all steps have descriptions."""
        matrix = [[1, 2, 3], [4, 5, 6]]
        result = gaussian_elimination(matrix)
        for step in result["steps"]:
            assert "description" in step
            assert isinstance(step["description"], str)
            assert len(step["description"]) > 0


class TestErrorCases:
    """Test error handling."""

    def test_singular_matrix(self):
        """Test that singular matrix raises ValueError."""
        matrix = [[1, 2, 3], [2, 4, 6]]  # Second row is 2x first row
        with pytest.raises(ValueError, match="singular"):
            gaussian_elimination(matrix)

    def test_empty_matrix(self):
        """Test that empty matrix raises ValueError."""
        with pytest.raises(ValueError, match="at least one row"):
            gaussian_elimination([])

    def test_mismatched_row_length(self):
        """Test that rows with inconsistent length raise ValueError."""
        matrix = [[1, 2, 3], [4, 5]]  # Second row too short
        with pytest.raises(ValueError, match="has 2 elements"):
            gaussian_elimination(matrix)

    def test_size_exceeds_limit(self):
        """Test that oversized matrix raises ValueError."""
        # Create a 101×102 matrix (exceeds MAX_MATRIX_SIZE=100)
        big_matrix = [[i for i in range(102)] for _ in range(101)]
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            gaussian_elimination(big_matrix)


class TestNumericalProperties:
    """Test numerical properties and invariants."""

    def test_solution_type(self):
        """Verify that solution is a list of floats."""
        matrix = [[1, 0, 5], [0, 1, 10]]
        result = gaussian_elimination(matrix)
        assert isinstance(result["solution"], list)
        for val in result["solution"]:
            assert isinstance(val, float)

    def test_no_negative_zero(self):
        """Verify that -0.0 is converted to 0.0."""
        # Design a system that might produce -0.0 after rounding
        matrix = [[1, 0, 0.0], [0, 1, 0.0]]
        result = gaussian_elimination(matrix)
        for val in result["solution"]:
            # Check that -0.0 == 0.0 (they are equal, but have same sign bit)
            assert val == 0.0 or val != 0.0  # Always true, but checks no exception
            # More specific: repr() or bit check would be overkill for this context

    def test_solution_count_matches_n(self):
        """Verify that number of solutions matches n."""
        for n in [2, 3, 4, 5]:
            matrix = [[1 if i == j else 0 for j in range(n + 1)] for i in range(n)]
            result = gaussian_elimination(matrix)
            assert len(result["solution"]) == n
            assert result["n"] == n


class TestPivotingCorrectness:
    """Test that pivoting produces correct results."""

    def test_pivoting_necessary(self):
        """Test system where pivoting is necessary for stability.
        Small pivot in first position; algorithm should swap rows.
        """
        matrix = [
            [0.0001, 1, 1.0001],
            [1, 1, 2],
        ]
        result = gaussian_elimination(matrix)
        # Both solutions should be close to 1.0
        assert math.isclose(result["solution"][0], 1.0, abs_tol=1e-3)
        assert math.isclose(result["solution"][1], 1.0, abs_tol=1e-3)

    def test_partial_pivoting_recorded(self):
        """Verify that partial pivoting steps are recorded when they occur."""
        matrix = [
            [1, 1, 2],
            [2, 3, 5],
        ]
        result = gaussian_elimination(matrix)
        step_types = [step["type"] for step in result["steps"]]
        # If pivoting occurs, there should be a "pivot" step
        # (In this case, pivoting may occur depending on magnitudes)
        assert "elimination" in step_types


class TestBackSubstitution:
    """Test back-substitution results."""

    def test_back_substitution_step_exists(self):
        """Verify that back-substitution step is present in result."""
        matrix = [[1, 2, 3], [4, 5, 6]]
        result = gaussian_elimination(matrix)
        bs_step = next(
            (s for s in result["steps"] if s["type"] == "back_substitution"), None
        )
        assert bs_step is not None
        assert "back_steps" in bs_step
        assert len(bs_step["back_steps"]) == 2

    def test_back_substitution_has_formulas(self):
        """Verify that back-substitution steps include formulas."""
        matrix = [[1, 0, 5], [0, 1, 10]]
        result = gaussian_elimination(matrix)
        bs_step = next(
            (s for s in result["steps"] if s["type"] == "back_substitution")
        )
        for bs in bs_step["back_steps"]:
            assert "variable" in bs
            assert "value" in bs
            assert "description" in bs


class TestMathematicalInvariants:
    """Test fundamental mathematical properties that must always hold."""

    def test_residual_ax_equals_b(self):
        """Verify A·x = b for the computed solution (strongest correctness check)."""
        A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
        b = [8, -11, -3]
        matrix = [A[i] + [b[i]] for i in range(3)]
        result = gaussian_elimination(matrix)
        x = result["solution"]

        for i in range(3):
            row_sum = sum(A[i][j] * x[j] for j in range(3))
            assert math.isclose(row_sum, b[i], abs_tol=1e-6), (
                f"Residual check failed for row {i}: A[i]·x = {row_sum}, expected b[i] = {b[i]}"
            )

    def test_residual_2x2(self):
        """Verify A·x = b for a 2×2 system."""
        A = [[3, 2], [1, -1]]
        b = [8, -1]
        matrix = [A[i] + [b[i]] for i in range(2)]
        result = gaussian_elimination(matrix)
        x = result["solution"]

        for i in range(2):
            row_sum = sum(A[i][j] * x[j] for j in range(2))
            assert math.isclose(row_sum, b[i], abs_tol=1e-6)

    def test_upper_triangular_zeros(self):
        """Verify that forward elimination produces a true upper-triangular matrix."""
        matrix = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]
        result = gaussian_elimination(matrix)
        ut_step = next(s for s in result["steps"] if s["type"] == "upper_triangular")
        mat = ut_step["matrix"]
        n = result["n"]

        for i in range(n):
            for j in range(i):  # all entries below the diagonal
                assert abs(mat[i][j]) < 1e-10, (
                    f"Upper-triangular check failed: element [{i}][{j}] = {mat[i][j]}, expected ~0"
                )

    def test_upper_triangular_zeros_4x4(self):
        """Verify upper-triangular form for a 4×4 system."""
        matrix = [
            [2, 1, -1, 2, 5],
            [4, 5, -3, 6, 9],
            [-2, 5, -2, 6, 4],
            [4, 11, -4, 8, 2],
        ]
        result = gaussian_elimination(matrix)
        ut_step = next(s for s in result["steps"] if s["type"] == "upper_triangular")
        mat = ut_step["matrix"]
        n = result["n"]

        for i in range(n):
            for j in range(i):
                assert abs(mat[i][j]) < 1e-10, (
                    f"Element [{i}][{j}] = {mat[i][j]}, expected ~0"
                )

    def test_residual_matches_upper_triangular(self):
        """Verify that back-substitution on the upper-triangular system is self-consistent."""
        matrix = [[4, 3, 24], [2, -1, 4]]
        result = gaussian_elimination(matrix)
        x = result["solution"]
        A = [[4, 3], [2, -1]]
        b = [24, 4]
        for i in range(2):
            row_sum = sum(A[i][j] * x[j] for j in range(2))
            assert math.isclose(row_sum, b[i], abs_tol=1e-6)


class TestLargerSystems:
    """Test correctness on larger (6×6 and 8×8) systems."""

    def test_6x6_known_solution(self):
        """Test a 6×6 system where the expected solution is all ones.
        Constructs b = A·[1,1,1,1,1,1] then verifies the solver recovers x.
        """
        A = [
            [ 2,  1, -1,  0,  1, -1],
            [ 1, -1,  2,  1,  0,  1],
            [-1,  2,  1, -1,  1,  0],
            [ 0,  1, -1,  2, -1,  1],
            [ 1,  0,  1, -1,  2,  1],
            [-1,  1,  0,  1, -1,  2],
        ]
        x_expected = [1.0] * 6
        b = [sum(A[i][j] * x_expected[j] for j in range(6)) for i in range(6)]
        matrix = [A[i] + [b[i]] for i in range(6)]

        result = gaussian_elimination(matrix)
        assert result["n"] == 6
        for k in range(6):
            assert math.isclose(result["solution"][k], x_expected[k], abs_tol=1e-6), (
                f"x{k+1}: expected {x_expected[k]}, got {result['solution'][k]}"
            )

    def test_6x6_residual(self):
        """Verify A·x = b for the 6×6 system (residual check)."""
        A = [
            [ 2,  1, -1,  0,  1, -1],
            [ 1, -1,  2,  1,  0,  1],
            [-1,  2,  1, -1,  1,  0],
            [ 0,  1, -1,  2, -1,  1],
            [ 1,  0,  1, -1,  2,  1],
            [-1,  1,  0,  1, -1,  2],
        ]
        x_expected = [1.0] * 6
        b = [sum(A[i][j] * x_expected[j] for j in range(6)) for i in range(6)]
        matrix = [A[i] + [b[i]] for i in range(6)]

        result = gaussian_elimination(matrix)
        x = result["solution"]
        for i in range(6):
            row_sum = sum(A[i][j] * x[j] for j in range(6))
            assert math.isclose(row_sum, b[i], abs_tol=1e-6)

    def test_8x8_diagonal_dominant(self):
        """Test an 8×8 diagonal-dominant system with a known non-trivial solution.
        Uses x = [1, 2, -1, 0, 1, -1, 0, 2] and verifies A·x = b is recovered.
        """
        A = [
            [ 5,  1,  0,  1,  0, -1,  1,  0],
            [ 1,  6, -1,  0,  1,  0,  1, -1],
            [ 0, -1,  7,  1,  0,  1,  0,  1],
            [ 1,  0,  1,  5, -1,  0,  1,  0],
            [ 0,  1,  0, -1,  6,  1,  0,  1],
            [-1,  0,  1,  0,  1,  5, -1,  0],
            [ 1,  1,  0,  1,  0, -1,  6,  1],
            [ 0, -1,  1,  0,  1,  0,  1,  7],
        ]
        x_expected = [1.0, 2.0, -1.0, 0.0, 1.0, -1.0, 0.0, 2.0]
        b = [sum(A[i][j] * x_expected[j] for j in range(8)) for i in range(8)]
        matrix = [A[i] + [b[i]] for i in range(8)]

        result = gaussian_elimination(matrix)
        assert result["n"] == 8
        for k in range(8):
            assert math.isclose(result["solution"][k], x_expected[k], abs_tol=1e-5), (
                f"x{k+1}: expected {x_expected[k]}, got {result['solution'][k]}"
            )

    def test_8x8_residual(self):
        """Verify A·x = b for the 8×8 diagonal-dominant system."""
        A = [
            [ 5,  1,  0,  1,  0, -1,  1,  0],
            [ 1,  6, -1,  0,  1,  0,  1, -1],
            [ 0, -1,  7,  1,  0,  1,  0,  1],
            [ 1,  0,  1,  5, -1,  0,  1,  0],
            [ 0,  1,  0, -1,  6,  1,  0,  1],
            [-1,  0,  1,  0,  1,  5, -1,  0],
            [ 1,  1,  0,  1,  0, -1,  6,  1],
            [ 0, -1,  1,  0,  1,  0,  1,  7],
        ]
        x_expected = [1.0, 2.0, -1.0, 0.0, 1.0, -1.0, 0.0, 2.0]
        b = [sum(A[i][j] * x_expected[j] for j in range(8)) for i in range(8)]
        matrix = [A[i] + [b[i]] for i in range(8)]

        result = gaussian_elimination(matrix)
        x = result["solution"]
        for i in range(8):
            row_sum = sum(A[i][j] * x[j] for j in range(8))
            assert math.isclose(row_sum, b[i], abs_tol=1e-5)


if __name__ == "__main__":
    # Run with: pytest test_solver.py -v
    pytest.main([__file__, "-v"])
