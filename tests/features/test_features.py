"""To run all tests: python -m unittest discover
To run only this test: python -m unittest tests.features.test_features"""

import unittest
import numpy as np
import tensorflow as tf
from project_name.features.featureimportance import (
    KNN_shap_graphs,
    NN_shap_graphs,
    averaged_NN_shap_graphs,
    averaged_NN_shap_graphs_per_output
)


class DummyModel(tf.Module):
    """Simple dummy model to simulate TF multi-output model."""

    def __call__(self, x, training=False):
        # Return a list of 3 tensors, each with shape (batch, 1)
        batch_size = tf.shape(x)[0]
        return [
            tf.ones((batch_size, 1)) * 0.1,
            tf.ones((batch_size, 1)) * 0.2,
            tf.ones((batch_size, 1)) * 0.3,
        ]


def dummy_build_model_fn(
    X_train, X_test,
    Y1_train, Y1_test,
    Y2_train, Y2_test,
    Y3_train, Y3_test,
    size_input
):
    """Dummy build model function that returns dummy model and scaler."""
    model = DummyModel()
    scaler = None
    val_acc1 = val_acc2 = val_acc3 = 0.9
    extra = None
    # Return expected signature from your code
    return model, X_train, X_test, scaler, val_acc1, val_acc2, val_acc3, extra


class TestShapFunctions(unittest.TestCase):
    def setUp(self):
        # Minimal dummy data for testing
        self.X_train = np.random.rand(1000, 5).astype(np.float32)
        self.X_test = np.random.rand(100, 5).astype(np.float32)
        self.Y1_train = np.zeros((1000,))
        self.Y1_test = np.zeros((100,))
        self.Y2_train = np.zeros((1000,))
        self.Y2_test = np.zeros((100,))
        self.Y3_train = np.zeros((1000,))
        self.Y3_test = np.zeros((100,))
        self.column_names = [f"feat_{i}" for i in range(5)]
        self.size_input = 5

    def test_knn_shap_graphs(self):
        # Define dummy predict_proba returning shape (n_samples, n_classes)
        def predict_proba(x):
            return np.ones((x.shape[0], 2)) * 0.5

        shap_values = KNN_shap_graphs(
            self.X_train,
            self.X_test,
            predict_proba,
            y_name="Dummy KNN",
            num_explain=5,
            column_names=self.column_names,
            plot=False,
        )
        self.assertEqual(
            shap_values.values.shape[0], 5,
            "SHAP values first dim should match num_explain"
        )

    def test_nn_shap_graphs(self):
        model = DummyModel()
        # Should run without error and no plot shown
        NN_shap_graphs(
            model,
            self.X_train,
            self.column_names,
            sample_size=3,
            no_explained=3,
            plot=False
            )

    def test_averaged_nn_shap_graphs(self):
        # Use dummy build_model_fn returning (model, scaler)
        def build_fn(*args, **kwargs):
            return DummyModel(), None

        averaged_NN_shap_graphs(
            build_fn,
            self.X_train, self.X_test,
            self.Y1_train, self.Y1_test,
            self.Y2_train, self.Y2_test,
            self.Y3_train, self.Y3_test,
            self.size_input,
            self.column_names,
            n_runs=2,
            sample_size=3,
            no_explained=3,
            plot=False
        )

    def test_averaged_nn_shap_graphs_per_output(self):
        averaged_NN_shap_graphs_per_output(
            dummy_build_model_fn,
            self.X_train, self.X_test,
            self.Y1_train, self.Y1_test,
            self.Y2_train, self.Y2_test,
            self.Y3_train, self.Y3_test,
            self.size_input,
            self.column_names,
            n_runs=2,
            sample_size=3,
            no_explained=3,
            plot=False,
        )


if __name__ == "__main__":
    unittest.main()
