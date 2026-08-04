import json
import math
import unittest

from topology_gate.rls import (
    MAX_RLS_FEATURES,
    MAX_RLS_SCHEDULE_LENGTH,
    RLS,
    RecursiveRidgeLeastSquares,
    RLSConfig,
)
from topology_gate.types import RLSState, RLSUpdate


def _solve(matrix, vector):
    """Small pivoted Gaussian solver used for independent test references."""

    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-14:
            raise AssertionError("singular reference system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _weighted_batch_solution(xs, ys, ridge, lambdas):
    size = len(xs[0])
    information = [
        [ridge if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]
    rhs = [0.0] * size
    for x, y, factor in zip(xs, ys, lambdas):
        information = [[factor * value for value in row] for row in information]
        rhs = [factor * value for value in rhs]
        for row in range(size):
            rhs[row] += x[row] * y
            for column in range(size):
                information[row][column] += x[row] * x[column]
    return _solve(information, rhs)


def _jacobi_minimum(matrix):
    work = [row[:] for row in matrix]
    size = len(work)
    for _ in range(max(20, 50 * size * size)):
        p, q = 0, 1
        largest = 0.0
        for row in range(size):
            for column in range(row + 1, size):
                if abs(work[row][column]) > largest:
                    p, q = row, column
                    largest = abs(work[row][column])
        if largest <= 1.0e-14:
            break
        app, aqq, apq = work[p][p], work[q][q], work[p][q]
        tau = (aqq - app) / (2.0 * apq)
        t = (1.0 if tau >= 0.0 else -1.0) / (
            abs(tau) + math.sqrt(1.0 + tau * tau)
        )
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        for index in range(size):
            if index in (p, q):
                continue
            aip, aiq = work[index][p], work[index][q]
            work[index][p] = work[p][index] = c * aip - s * aiq
            work[index][q] = work[q][index] = s * aip + c * aiq
        work[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        work[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        work[p][q] = work[q][p] = 0.0
    return min(row[index] for index, row in enumerate(work))


class RecursiveRidgeLeastSquaresTests(unittest.TestCase):
    def assertVectorClose(self, left, right, places=12):
        self.assertEqual(len(left), len(right))
        for actual, expected in zip(left, right):
            self.assertAlmostEqual(actual, expected, places=places)

    def assertMatrixClose(self, left, right, places=12):
        self.assertEqual(len(left), len(right))
        for actual_row, expected_row in zip(left, right):
            self.assertVectorClose(actual_row, expected_row, places=places)

    def test_constant_lambda_equivalence(self):
        xs = [[1.0, -1.0], [0.5, 2.0], [-2.0, 0.25], [1.5, 1.0]]
        ys = [2.0, -0.5, 1.25, 3.0]
        factor = 0.91
        ridge = 0.7

        configured = RLS(2, ridge=ridge, forgetting_factor=factor)
        scheduled = RecursiveRidgeLeastSquares(
            2, ridge=ridge, forgetting_factor=[factor] * len(xs)
        )
        overridden = RLS(2, ridge=ridge)
        for x, y in zip(xs, ys):
            configured.update(x, y)
            scheduled.update(x, y)
            overridden.update(x, y, forgetting_factor=factor)

        self.assertVectorClose(configured.coefficients, scheduled.coefficients)
        self.assertVectorClose(configured.coefficients, overridden.coefficients)
        self.assertMatrixClose(configured.covariance, scheduled.covariance)
        self.assertMatrixClose(configured.covariance, overridden.covariance)
        self.assertVectorClose(
            configured.coefficients,
            _weighted_batch_solution(xs, ys, ridge, [factor] * len(xs)),
        )

    def test_adaptive_forgetting_matches_weighted_batch(self):
        xs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]
        ys = [1.0, 2.0, 2.75, 0.5]
        factors = [1.0, 0.8, 0.95, 0.72]
        model = RLS(2, ridge=0.4, forgetting_factor=factors)

        for x, y in zip(xs, ys):
            model.update(x, y)

        self.assertVectorClose(
            model.coefficients,
            _weighted_batch_solution(xs, ys, 0.4, factors),
        )
        self.assertEqual(model.forgetting_factors, tuple(factors))
        self.assertAlmostEqual(model.last_forgetting_factor, factors[-1])

        callback_model = RLS(
            2,
            ridge=0.4,
            forgetting_factor=lambda step: factors[step - 1],
        )
        for x, y in zip(xs, ys):
            receipt = callback_model.update(x, y)
            self.assertIsInstance(receipt, RLSUpdate)
            self.assertIsInstance(receipt.state, RLSState)
        self.assertVectorClose(model.coefficients, callback_model.coefficients)

        label_independent = RLS(
            2,
            lambda_min=0.7,
            lambda_max=0.95,
            forgetting_factor=lambda step: 0.9 if step % 2 else 0.8,
        )
        for x, y in zip(xs, ys):
            label_independent.update(x, y)
        self.assertEqual(label_independent.forgetting_factors, (0.9, 0.8, 0.9, 0.8))
        self.assertEqual(
            RLSConfig(
                n_features=2,
                lambda_min=0.7,
                lambda_max=0.95,
                forgetting_factor=0.9,
            ).lambda_max,
            0.95,
        )

    def test_multi_output_contract_and_state_round_trip(self):
        model = RLS(2, n_outputs=2, ridge=0.5, forgetting_factor=0.9)
        receipt = model.update([1.0, 0.0], [1.0, 2.0])

        self.assertEqual(receipt.prediction, [0.0, 0.0])
        self.assertEqual(receipt.residual, [1.0, 2.0])
        self.assertEqual(len(receipt.state.coefficients), 2)
        self.assertEqual(len(receipt.state.coefficients[0]), 2)
        self.assertEqual(model.predict([0.0, 1.0]), [0.0, 0.0])

        restored = RLS.from_state_dict(model.state_dict())
        self.assertEqual(restored.predict([1.0, 0.0]), model.predict([1.0, 0.0]))
        with self.assertRaises(ValueError):
            model.update([1.0, 0.0], 1.0)

    def test_reset_restores_ridge_prior_and_rewinds_schedule(self):
        model = RLS(2, ridge=2.0, forgetting_factor=[0.8, 0.6])
        first = model.update([1.0, 0.0], 3.0)
        self.assertEqual(model.n_updates, 1)
        model.reset()

        self.assertEqual(model.n_updates, 0)
        self.assertEqual(model.coefficients, [0.0, 0.0])
        self.assertEqual(model.covariance, [[0.5, 0.0], [0.0, 0.5]])
        self.assertEqual(model.forgetting_factors, ())
        self.assertIsNone(model.last_prediction)
        self.assertIsNone(model.last_residual)
        self.assertEqual(model.update([1.0, 0.0], 3.0), first)
        self.assertEqual(model.last_forgetting_factor, 0.8)

    def test_dimensions_and_batch_prediction(self):
        model = RLS(2)
        self.assertEqual(model.predict([0.0, 0.0]), 0.0)
        self.assertEqual(model.predict([[1.0, 2.0], [3.0, 4.0]]), [0.0, 0.0])
        self.assertEqual(len(model.update([1.0, 2.0], 1.0).state.coefficients), 2)

        with self.assertRaises(ValueError):
            model.predict([1.0])
        with self.assertRaises(ValueError):
            model.predict([[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            model.update([[1.0, 2.0]], 1.0)
        with self.assertRaises(ValueError):
            model.update([1.0], 1.0)

    def test_covariance_is_symmetric_and_psd_with_tolerance(self):
        model = RLS(3, ridge=1.0e-3, forgetting_factor=0.83)
        observations = [
            ([1.0, 1.0, 1.0], 1.0),
            ([1.0, 1.0 + 1.0e-10, 1.0 - 1.0e-10], 1.1),
            ([1000.0, -1000.0, 1.0], -2.0),
            ([1.0e-6, 2.0e-6, -3.0e-6], 0.25),
        ] * 15
        for x, y in observations:
            model.update(x, y)

        covariance = model.covariance
        scale = max(abs(value) for row in covariance for value in row)
        for row in range(3):
            for column in range(3):
                self.assertAlmostEqual(
                    covariance[row][column], covariance[column][row], places=14
                )
        self.assertGreaterEqual(_jacobi_minimum(covariance), -1.0e-10 * max(1.0, scale))

    def test_state_round_trip_is_deterministic(self):
        model = RLS(2, ridge=0.25, forgetting_factor=0.9)
        for x, y in [([1.0, 2.0], 1.0), ([2.0, -1.0], 3.0)]:
            model.update(x, y)
        state = model.state_dict()
        self.assertEqual(state, model.get_state())
        self.assertEqual(json.dumps(state, sort_keys=True), json.dumps(model.state_dict(), sort_keys=True))

        restored = RLS.from_state_dict(state)
        restored_json = RLS.from_json(model.to_json())
        for candidate in (restored, restored_json):
            self.assertVectorClose(candidate.coefficients, model.coefficients)
            self.assertMatrixClose(candidate.covariance, model.covariance)
            candidate.update([0.5, 0.25], 2.0)
        model.update([0.5, 0.25], 2.0)
        self.assertVectorClose(restored.coefficients, model.coefficients)
        self.assertVectorClose(restored_json.coefficients, model.coefficients)

    def test_invalid_inputs_are_rejected(self):
        for kwargs in (
            {"n_features": 0},
            {"n_features": 2, "ridge": 0.0},
            {"n_features": 2, "ridge": -1.0},
            {"n_features": 2, "forgetting_factor": 0.0},
            {"n_features": 2, "forgetting_factor": -0.1},
            {"n_features": 2, "forgetting_factor": 1.01},
            {"n_features": 2, "forgetting_factor": math.nan},
            {"n_features": 2, "forgetting_factor": math.inf},
            {"n_features": 2, "forgetting_factor": []},
        ):
            with self.assertRaises((TypeError, ValueError)):
                RLS(**kwargs)

        model = RLS(2)
        for x, y in (([math.nan, 1.0], 1.0), ([1.0, math.inf], 1.0), ([1.0, 2.0], math.nan)):
            with self.assertRaises((ValueError, FloatingPointError)):
                model.update(x, y)
        with self.assertRaises(ValueError):
            model.update([1.0, 2.0], 1.0, lambda_t=0.0)
        with self.assertRaises(ValueError):
            model.update([1.0, 2.0], 1.0, lambda_=0.9, lambda_t=0.8)

        exhausted = RLS(1, forgetting_factor=[0.9])
        exhausted.update([1.0], 1.0)
        with self.assertRaises(ValueError):
            exhausted.update([1.0], 1.0)

        bad_state = model.state_dict()
        original_state = model.state_dict()
        bad_state["theta"] = [math.nan, 0.0]
        with self.assertRaises(ValueError):
            model.set_state(bad_state)
        self.assertEqual(model.state_dict(), original_state)

        non_psd = model.state_dict()
        non_psd["covariance"] = [[1.0, 2.0], [2.0, 1.0]]
        with self.assertRaises(ValueError):
            model.set_state(non_psd)

        with self.assertRaises(ValueError):
            RLS(2, lambda_min=0.9, lambda_max=0.8)
        with self.assertRaises(ValueError):
            RLS(2, lambda_min=0.9, lambda_max=0.95, forgetting_factor=0.8)

        with self.assertRaises(ValueError):
            RLS(MAX_RLS_FEATURES + 1)
        with self.assertRaises(ValueError):
            RLS(1, forgetting_factor=[0.9] * (MAX_RLS_SCHEDULE_LENGTH + 1))


if __name__ == "__main__":
    unittest.main()
