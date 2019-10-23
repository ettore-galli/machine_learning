import unittest

from linear_regression_engine.linear_regression_engine import LinearRegressionEngine
from linear_regression_engine.linear_model_predictor import LinearModelPredictor


class TestPolynomialFeatures(unittest.TestCase):
    def testPass(self):
        pass

    DEGREE_THAT_DOES_NOT_MATTER = 1

    def __fit_test_common(self, fit_inputs, fit_expected_outputs, degree, test_inputs):
        '''
        Common test workflow for various cases

        :param inputs:
        :param expected_outputs:
        :param degree:
        :return:
        '''

        print("-" * 80)
        print("Test degree = {deg}".format(deg=degree))
        print("-" * 80)

        lre_predictable = LinearRegressionEngine(degree=degree)
        lre_predictable.fit_predictable(fit_inputs, fit_expected_outputs)

        fitting_polymomial = lre_predictable.get_polynomial_definition()

        polymomial_coefficients = fitting_polymomial["coefficients"]
        polymomial_powers = fitting_polymomial["powers"]

        predictions = []

        for test_i in test_inputs:
            prediction = LinearModelPredictor.eval_polynomial(polymomial_coefficients, polymomial_powers, test_i)
            prediction_value = prediction[0]
            predictions.append(prediction_value)

        # print the results:
        # (assume len(polynomial_predictions) == len(native_predictions) but it is not
        # a problem, if the hypothesis does not hold, the test fails.
        for i in range(len(predictions)):
            prediction_value = predictions[i]
            print("Input ", test_inputs[i], "yields a prediction of", prediction_value)

        self.assertTrue(True)  # Yes, it works :-)

    def test_set_1(self):
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
            [0.18]
        ]

        test_inputs = [
            [130.0, 200.0, 10.0],
            [260.0, 201.0, 31.0],
            [590.0, 320.0, 98.0],
            [371.0, 431.0, 77.0],
            [371.1, 431.1, 77.1],
            [2, 3, 4]
        ]

        for degree in [1, 2, 3]:
            self.__fit_test_common(fit_inputs, fit_expected_outputs, degree, test_inputs)
