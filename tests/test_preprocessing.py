###
## cluster_maker - test file for preprocessing module
## Tests for feature selection and standardisation
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):

    # Test 1: Detect non-numeric columns that would crash downstream algorithms
    def test_select_features_rejects_non_numeric_columns(self):
        """
        Real problem: If a user accidentally includes a categorical or string column
        in their feature list, the clustering algorithm will crash with cryptic numpy errors.
        This test ensures select_features() catches this early with a clear, actionable error
        message before data reaches the algorithms layer.
        """
        data = pd.DataFrame({
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
            "category": ["A", "B", "C"],  # Non-numeric column
        })

        # Should work fine with purely numeric columns
        result = select_features(data, ["feature_a", "feature_b"])
        self.assertEqual(result.shape, (3, 2))
        self.assertListEqual(list(result.columns), ["feature_a", "feature_b"])

        # Should raise TypeError when trying to include categorical column
        with self.assertRaises(TypeError) as context:
            select_features(data, ["feature_a", "category"])
        self.assertIn("not numeric", str(context.exception).lower())

    # Test 2: Verify standardisation produces mathematically correct output
    def test_standardise_features_produces_zero_mean_unit_variance(self):
        """
        Real problem: If standardisation fails (wrong axis, missing scaling, etc.),
        downstream clustering will be biased. Features with larger ranges will dominate
        distance calculations, and K-means will produce meaningless results biased toward
        high-variance dimensions. This test verifies the statistical properties are correct.
        """
        # Create data with wildly different scales to expose scaling bugs
        X = np.array([
            [1.0, 100.0],
            [2.0, 200.0],
            [3.0, 300.0],
            [4.0, 400.0],
            [5.0, 500.0],
        ], dtype=float)

        X_scaled = standardise_features(X)

        # After standardisation, each feature should have mean ≈ 0 (within floating-point tolerance)
        means = X_scaled.mean(axis=0)
        np.testing.assert_array_almost_equal(means, [0.0, 0.0], decimal=10,
                                             err_msg="Means not zero after standardisation")

        # After standardisation, each feature should have variance ≈ 1
        # (StandardScaler uses population variance with ddof=0)
        variances = X_scaled.var(axis=0, ddof=0)
        np.testing.assert_array_almost_equal(variances, [1.0, 1.0], decimal=10,
                                             err_msg="Variances not unit after standardisation")

        # Additional check: verify shape is preserved
        self.assertEqual(X_scaled.shape, X.shape,
                        "Standardisation changed data shape")

    # Test 3: Verify missing column detection prevents silent failures
    def test_select_features_raises_clear_error_for_missing_columns(self):
        """
        Real problem: If a user requests a column that doesn't exist (typo, wrong dataset, etc.),
        pandas might silently fail or produce cryptic KeyError at runtime. This test ensures
        select_features() detects ALL missing columns upfront and reports them clearly,
        making debugging fast and obvious rather than hunting through error traces.
        """
        data = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "z": [7.0, 8.0, 9.0],
        })

        # Request columns: 2 valid, 2 missing
        with self.assertRaises(KeyError) as context:
            select_features(data, ["x", "y", "missing_col_1", "missing_col_2"])

        error_msg = str(context.exception)
        # Verify BOTH missing columns are reported
        self.assertIn("missing_col_1", error_msg,
                     "First missing column not mentioned in error")
        self.assertIn("missing_col_2", error_msg,
                     "Second missing column not mentioned in error")
        self.assertIn("missing", error_msg.lower(),
                     "Error doesn't clearly indicate 'missing' problem")

        # Also test single missing column case
        with self.assertRaises(KeyError):
            select_features(data, ["x", "nonexistent"])


if __name__ == "__main__":
    unittest.main()
