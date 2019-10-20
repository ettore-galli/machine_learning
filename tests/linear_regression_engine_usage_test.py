import unittest

from linear_regression_engine.linear_regression_engine import LinearRegressionEngine
from linear_regression_engine.linear_model_predictor import LinearModelPredictor


class TestPolynomialFeatures(unittest.TestCase):
    def testPass(self):
        pass

    DEGREE_THAT_DOES_NOT_MATTER = 1

    def __native_versus_predictable_test_common(self, fit_inputs, fit_expected_outputs, degree, test_inputs):
        '''
        Common test workflow for various cases

        :param inputs:
        :param expected_outputs:
        :param degree:
        :return:
        '''

        # Use native predictor

        lre_native = LinearRegressionEngine(degree=degree)
        lre_native.fit_native(fit_inputs, fit_expected_outputs)

        native_predictions = []
        for test_i in test_inputs:
            prediction = lre_native.predict_native(test_i)
            prediction_value = prediction[0][0]
            native_predictions.append(prediction[0][0])

        # Use "predictable" predictor (i.e. predictable polynomial features)
        lre_predictable = LinearRegressionEngine(degree=degree)
        lre_predictable.fit_predictable(fit_inputs, fit_expected_outputs)

        fitting_polymomial = lre_predictable.get_polynomial_definition()
        polymomial_coefficients = fitting_polymomial["coefficients"]
        polymomial_powers = fitting_polymomial["powers"]

        polynomial_predictions = []
        for test_i in test_inputs:
            prediction = LinearModelPredictor.eval_polynomial(polymomial_coefficients, polymomial_powers, test_i)
            prediction_value = prediction[0]
            polynomial_predictions.append(prediction_value)

        # Compare the results:
        # (assume len(polynomial_predictions) == len(native_predictions) but it is not
        # a problem, if the hypothesis does not hold, the test fails.
        for i in range(len(native_predictions)):
            native_value = native_predictions[i]
            polymomial_value = polynomial_predictions[i]
            print ("Comparing ", native_value, "to", polymomial_value)
            self.assertAlmostEqual(native_value, polymomial_value)

    def test_native_versus_predictable_1(self):
        fit_inputs = [
            [200.0, 300.0, 50.0],
            [250.0, 200.0, 40.0],
            [350.0, 190.0, 55.0],
            [345.0, 180.0, 12.0],
            [371.0, 431.0, 77.0],
            [2.0, 3.0, 4.0]
        ]

        fit_expected_outputs = [
            [7.0],
            [7.4],
            [7.9],
            [14.1],
            [17.2],
            [0.1]
        ]

        test_inputs = [
            [130.0, 200.0, 10.0],
            [260.0, 201.0, 31.0],
            [590.0, 320.0, 98.0],
            [371.0, 431.0, 77.0],
            [371.1, 431.1, 77.1],
            [2, 3, 4]
        ]

        degree = 2

        self.__native_versus_predictable_test_common(fit_inputs, fit_expected_outputs, degree, test_inputs)
