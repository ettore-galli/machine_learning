import unittest

from linear_regression_engine.linear_regression_engine import LinearRegressionEngine


class TestPolynomialFeatures(unittest.TestCase):
    def testPass(self):
        pass

    DEGREE_THAT_DOES_NOT_MATTER = 1

    def test_merge_as_tuple(self):
        lre = LinearRegressionEngine(self.DEGREE_THAT_DOES_NOT_MATTER)
        self.assertEqual(
            lre.merge_as_tuple("a", ("b", "c")),
            ("a", "b", "c")
        )

    def test_cartesian_product_simple(self):
        lre = LinearRegressionEngine(self.DEGREE_THAT_DOES_NOT_MATTER)
        a = ("a", "b")
        b = ("x", "y")
        p = [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")]
        cp = lre.cartesian_product(a, b)
        self.assertEqual(p, cp)

    def test_tuple_product(self):
        lre = LinearRegressionEngine(self.DEGREE_THAT_DOES_NOT_MATTER)
        t = (1, 3, 4, 2)
        tp_calc = lre.tuple_product(t)
        tp_expected = 24
        self.assertEqual(tp_calc, tp_expected)

    def test_predictable_polynomial_features_degree_1(self):
        lre = LinearRegressionEngine(self.DEGREE_THAT_DOES_NOT_MATTER)
        variables = ["a", "b", "c"]
        ppf_calc = lre.predictable_polynomial_features(variables, 1)
        ppf_expected = ['1', 'a', 'b', 'c']
        self.assertEqual(ppf_calc, ppf_expected)

    def test_predictable_polynomial_features_degree_2(self):
        lre = LinearRegressionEngine(self.DEGREE_THAT_DOES_NOT_MATTER)
        variables = ["a", "b", "c"]
        ppf_calc = lre.predictable_polynomial_features(variables, 2)
        ppf_expected = [
            ('1', '1'),
            ('1', 'a'),
            ('1', 'b'),
            ('1', 'c'),
            ('a', 'a'),
            ('a', 'b'),
            ('a', 'c'),
            ('b', 'b'),
            ('b', 'c'),
            ('c', 'c')
        ]

        self.assertEqual(ppf_calc, ppf_expected)

    def test_predictable_polynomial_features_degree_3(self):
        lre = LinearRegressionEngine(self.DEGREE_THAT_DOES_NOT_MATTER)
        variables = ["a", "b", "c"]
        ppf_calc = lre.predictable_polynomial_features(variables, 3)
        ppf_expected = [
            ('1', '1', '1'),
            ('1', '1', 'a'),
            ('1', '1', 'b'),
            ('1', '1', 'c'),
            ('1', 'a', 'a'),
            ('1', 'a', 'b'),
            ('1', 'a', 'c'),
            ('1', 'b', 'b'),
            ('1', 'b', 'c'),
            ('1', 'c', 'c'),
            ('a', 'a', 'a'),
            ('a', 'a', 'b'),
            ('a', 'a', 'c'),
            ('a', 'b', 'b'),
            ('a', 'b', 'c'),
            ('a', 'c', 'c'),
            ('b', 'b', 'b'),
            ('b', 'b', 'c'),
            ('b', 'c', 'c'),
            ('c', 'c', 'c')
        ]
        self.assertEqual(ppf_calc, ppf_expected)

    def test_predictable_polynomial_powers_degree_2(self):
        lre = LinearRegressionEngine(self.DEGREE_THAT_DOES_NOT_MATTER)
        variables = ["a", "b", "c"]
        ppw_calc = lre.predictable_polynomial_powers(variables, 2)
        ppw_expected = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0],
            [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]
        ]

        self.assertEqual(ppw_calc, ppw_expected)

    def test_predictable_polynomial_powers_degree_3(self):
        lre = LinearRegressionEngine(self.DEGREE_THAT_DOES_NOT_MATTER)
        variables = ["a", "b", "c"]
        ppw_calc = lre.predictable_polynomial_powers(variables, 3)
        ppw_expected = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0],
            [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2],
            [3, 0, 0], [2, 1, 0], [2, 0, 1], [1, 2, 0], [1, 1, 1],
            [1, 0, 2], [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3]
        ]

        self.assertEqual(ppw_calc, ppw_expected)

    def test_eval_polynomial_terms(self):
        lre = LinearRegressionEngine(self.DEGREE_THAT_DOES_NOT_MATTER)
        polynomial_powers = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0],
            [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]
        ]
        polynomial_coefficients = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        x = [1, 2, 3]
        pt_calc = lre.eval_polynomial_terms(polynomial_powers, polynomial_coefficients, x)
        pt_expected = [0.1, 0.2, 0.6, 1.2, 0.5, 1.2, 2.1, 3.2, 5.4, 9.0]
        for actual, expected in zip(pt_calc, pt_expected):
            self.assertAlmostEqual(actual, expected, delta=0.001)

    def test_create_polynomial_features(self):
        lre = LinearRegressionEngine(self.DEGREE_THAT_DOES_NOT_MATTER)
        polynomial_powers = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0],
            [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]
        ]
        x = [1, 2, 3]
        pf_calc = lre.create_polynomial_features(polynomial_powers, x)
        pf_expected = [1, 1, 2, 3, 1, 2, 3, 4, 6, 9]
        for actual, expected in zip(pf_calc, pf_expected):
            self.assertAlmostEqual(actual, expected, delta=0.000001)
