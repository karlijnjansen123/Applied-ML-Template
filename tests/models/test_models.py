import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from project_name.models.KNN import KNN_solver
from project_name.models.NeuralNetwork import (test_train_split,
                                               build_neural_network)


class TestKNNSolver(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.random.rand(100, 5))
        self.y = pd.Series(
            np.random.randint(0, 2, size=100),
            name="dummy_target"
            )

    def test_knn_solver_accuracy_range(self):
        score, X_train, X_test, predict_proba = KNN_solver(
            self.X,
            self.y,
            scoring="accuracy",
            plot=False
            )
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_knn_solver_shapes(self):
        score, X_train, X_test, predict_proba = KNN_solver(
            self.X,
            self.y,
            scoring="accuracy",
            plot=False
            )
        self.assertEqual(X_train.shape[1], self.X.shape[1])
        self.assertEqual(X_test.shape[1], self.X.shape[1])


class TestTrainSplit(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.random.rand(100, 10))
        self.Y1 = pd.Series(np.random.randint(1, 6, 100))
        self.Y2 = pd.Series(np.random.randint(1, 6, 100))
        self.Y3 = pd.Series(np.random.randint(1, 6, 100))

    def test_output_shapes(self):
        splits = test_train_split(self.X, self.Y1, self.Y2, self.Y3)
        (X_train, X_test,
         Y1_train, Y1_test,
         Y2_train, Y2_test,
         Y3_train, Y3_test) = splits

        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(Y1_train), 80)
        self.assertEqual(len(Y1_test), 20)
        self.assertEqual(len(Y2_train), 80)
        self.assertEqual(len(Y2_test), 20)
        self.assertEqual(len(Y3_train), 80)
        self.assertEqual(len(Y3_test), 20)

    def test_return_types(self):
        splits = test_train_split(self.X, self.Y1, self.Y2, self.Y3)
        self.assertEqual(len(splits), 8)
        for part in splits:
            self.assertTrue(isinstance(part, (pd.DataFrame, pd.Series)))


class TestBuildNeuralNetwork(unittest.TestCase):
    def setUp(self):
        # Create mock data for testing
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.rand(100, 10))
        self.Y1 = pd.Series(np.random.randint(1, 6, size=100))
        self.Y2 = pd.Series(np.random.randint(1, 6, size=100))
        self.Y3 = pd.Series(np.random.randint(1, 6, size=100))
        (self.X_train, self.X_test, self.Y1_train, self.Y1_test,
         self.Y2_train, self.Y2_test, self.Y3_train,
         self.Y3_test) = train_test_split(
            self.X, self.Y1, self.Y2, self.Y3, test_size=0.2, random_state=42
        )

    def test_build_neural_network_output(self):
        (model, X_train_scaled, X_test_scaled, scaler, val_acc1,
         val_acc2, val_acc3, metrics_dict) = build_neural_network(
            self.X_train, self.X_test,
            self.Y1_train, self.Y1_test,
            self.Y2_train, self.Y2_test,
            self.Y3_train, self.Y3_test,
            size_input=self.X.shape[1]
        )

        self.assertIsNotNone(model)
        self.assertEqual(X_train_scaled.shape[0], self.X_train.shape[0])
        self.assertEqual(X_test_scaled.shape[0], self.X_test.shape[0])
        self.assertTrue(0 <= val_acc1 <= 1)
        self.assertTrue(0 <= val_acc2 <= 1)
        self.assertTrue(0 <= val_acc3 <= 1)
        self.assertIn("think_body", metrics_dict)
        self.assertIn("feeling_low", metrics_dict)
        self.assertIn("sleep_difficulty", metrics_dict)
        self.assertIn("f1_score", metrics_dict["think_body"])
        self.assertIn("auc_score", metrics_dict["think_body"])


if __name__ == "__main__":
    unittest.main()
