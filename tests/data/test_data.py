"""To run all tests: python -m unittest discover
To run only this test: python -m unittest tests.data.test_data"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from project_name.data.preprocessing import preprocess_hbsc_data


class PreprocessTest(unittest.TestCase):

    def create_temp_csv(self, df):
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, "test.csv")
        df.to_csv(file_path, sep=";", index=False)
        return file_path

    def test_replace_illegal_values_with_nan(self):
        df = pd.DataFrame({
            'col1': [1, 99, '999', ' ', -99.0],  # 3 illegal values
            'col2': [2, -999, 3, 999.0, '   '],  # 3 illegal values
            'y': [1, 2, 3, 4, 5]
        })
        file_path = self.create_temp_csv(df)

        processed = preprocess_hbsc_data(
            file_path,
            selected_columns=['col1', 'col2'],
            emc_cols=[], y=['y']
            )

        # NaNs should have been imputed — so there should be 0 NaNs
        self.assertEqual(processed['col1'].isnull().sum(), 0)
        self.assertEqual(processed['col2'].isnull().sum(), 0)

        # Confirm values have changed due to imputation (not all original)
        # For instance, median of [1, 1, 1] would be used to fill NaNs
        imputed_values_col1 = processed['col1'].value_counts().to_dict()
        self.assertGreaterEqual(imputed_values_col1.get(1.0, 0), 1)

        # Confirm type is numeric
        self.assertTrue(np.issubdtype(processed['col1'].dtype, np.number))

    def test_convert_to_numeric_and_coerce_errors(self):
        df = pd.DataFrame({
            'col1': ['5', 'non-numeric', '7'],
            'col2': ['10', '20', 'error'],
            'y': [1, 2, 3]
        })
        file_path = self.create_temp_csv(df)

        processed = preprocess_hbsc_data(
            file_path,
            selected_columns=['col1', 'col2'],
            emc_cols=[], y=['y']
            )

        # Check that col1 and col2 are numeric
        self.assertTrue(np.issubdtype(processed['col1'].dtype, np.number))
        self.assertTrue(np.issubdtype(processed['col2'].dtype, np.number))

        # Check no NaNs remain due to imputation
        self.assertEqual(processed['col1'].isnull().sum(), 0)
        self.assertEqual(processed['col2'].isnull().sum(), 0)

        # Check that the non-numeric string got replaced (e.g. median imputed)
        expected_median = np.median([5, 7])
        self.assertIn(expected_median, processed['col1'].values)

    def test_emc_columns_summed_correctly(self):
        df = pd.DataFrame({
            'emc1': [1, 2, np.nan],
            'emc2': [3, np.nan, 4],
            'col1': [0, 0, 0],
            'y': [1, 2, 3]
        })
        file_path = self.create_temp_csv(df)

        processed = preprocess_hbsc_data(
            file_path,
            selected_columns=['col1'],
            emc_cols=['emc1', 'emc2'],
            y=['y']
            )

        expected = [4, 2, 4]
        np.testing.assert_array_almost_equal(
            processed['emcsocmed_sum'].values, expected
            )

    def test_impute_median_on_selected_columns(self):
        df = pd.DataFrame({
            'col1': [1, np.nan, 3, np.nan, 5],
            'col2': [10, 20, 30, 40, 50],
            'y': [1, 2, 3, 4, 5]
        })
        file_path = self.create_temp_csv(df)

        processed = preprocess_hbsc_data(
            file_path,
            selected_columns=['col1', 'col2'],
            emc_cols=[],
            y=['y']
            )

        self.assertTrue(processed['col1'].isnull().sum() == 0)
        self.assertIn(df['col1'].median(), processed['col1'].values)

    def test_drop_rows_with_nan_in_target(self):
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'y': [1, np.nan, 3]
        })
        file_path = self.create_temp_csv(df)

        processed = preprocess_hbsc_data(
            file_path,
            selected_columns=['col1', 'col2'],
            emc_cols=[],
            y=['y']
            )

        self.assertEqual(processed.shape[0], 2)
        self.assertEqual(processed['y'].isnull().sum(), 0)

    def test_impute_emcsocmed_sum_median(self):
        df = pd.DataFrame({
            'emc1': [np.nan, 2, 3],
            'emc2': [np.nan, 3, 1],
            'col1': [0, 0, 0],
            'y': [1, 2, 3]
        })
        file_path = self.create_temp_csv(df)

        processed = preprocess_hbsc_data(
            file_path,
            selected_columns=['col1'],
            emc_cols=['emc1', 'emc2'],
            y=['y']
            )

        self.assertEqual(processed['emcsocmed_sum'].isnull().sum(), 0)
        self.assertTrue(np.issubdtype(
            processed['emcsocmed_sum'].dtype,
            np.number
            ))


if __name__ == '__main__':
    unittest.main()
