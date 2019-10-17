import unittest


from linear_regression_engine.linear_regression_engine import LinearRegressionEngine

class TestPolynomialFeatures(unittest.TestCase):
    def testPass(self):
        pass


    def test_merge_as_tuple(self):
        lre = LinearRegressionEngine(1)
        self.assertEqual(
            lre.merge_as_tuple("a", ("b", "c")),
            ("a", "b", "c")
        )

if __name__ == '__main__':
    unittest.main()