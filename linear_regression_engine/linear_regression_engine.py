import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata
from sklearn import linear_model, pipeline, preprocessing
from typing import TypeVar, MutableSequence, List, Iterable


class LinearRegressionEngine:
    '''
    Wraps the scikit-learn polynomial least squares regression algorithm
    with the purpose of allowing the definition of the polynomial regressor
    to be taken outside and used without the scikit-learn library and environment
    '''
    POLYNOMIAL_COEFFICIENTS_KEY = "coefficients"
    POLYNOMIAL_POWERS_KEY = "powers"

    PolymorphicItem = TypeVar('ListElement', int, float, str)
    NumericItem = TypeVar('ListElement', int, float)

    # x:MutableSequence[SymbolicListElement]

    def __init__(self, degree: int = 1, normalize: bool = False, n_jobs: int = None):
        '''
        Contructor
        :param degree: Degree of the polynomial.
        :param normalize: linear_model parameter normalize; TODO: Verify if applies
        :param n_jobs: linear_model parameter n_jobs
        '''
        self.fit_intercept = False
        self.copy_X = True
        self.degree = degree
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.model = None
        self.number_of_variables = None
        self.polynomial_powers = None
        self.poly_def = None

    def merge_as_tuple(self, a, b):
        '''
        Merges a single value and a tuple or two tuples into a single tuple

        :param a: Tuple or value
        :param b: Tuple or value
        :return: Merged tuple
        '''
        a_t = a if type(a) is tuple else (str(a),)
        b_t = b if type(b) is tuple else (str(b),)
        return a_t + b_t

    def cartesian_product(self, x: List[PolymorphicItem], y: List[PolymorphicItem]):
        '''
        Cartesian product of two "sets" **without** repetition.

        Given two "sets" (i.e. lists with no assumption on repetitions, although
        it is assumed that there aren't) performs the cartesian product, giving
        a list of tuples each one representing the single term of the product

        :param x:
        :param y:
        :return: The cartesian product
        '''
        cp = []
        for xi in x:
            for yi in y:
                px = tuple(sorted(self.merge_as_tuple(xi, yi)))
                if px not in cp:
                    cp.append(px)
        return cp

    def tuple_product(self, tp: List[NumericItem]):
        '''
        Calculates the product of all items in the given tuple, each one assumed to
        be numeric.

        :param tp: A tuple of numbers
        :return: The product of tuple elements
        '''
        if len(tp) == 0:
            return 1
        else:
            return tp[0] * self.tuple_product(tp[1:])

    def predictable_polynomial_features(self, x: List[PolymorphicItem], degree: int = 1):
        '''
        Given a set of symbolic variables, creates the combination of variables
        that make up a polynomial of the given degree.

        Multiple cartesian product up to the given degree

        :param x: A set of symbols representing variables
        :param degree: The degree of the polynpmial
        :return: The iterated cross cartesian product up to the desired degree
        '''
        if degree == 1:
            return [(str(k)) for k in [1] + x]
        else:
            ppf = [1] + x
            for i in range(1, degree):
                ppf = self.cartesian_product(ppf, [1] + x)

            return ppf

    def predictable_polynomial_powers(self, x_symbols: List[PolymorphicItem], degree: int = 1):
        '''
        Calculates the powers of each variable for each term of the polynomial

        :param x_symbols: Variable symbols
        :param degree: Polynomial degree
        :return: List of list of powers of the corresponding variables
        '''
        polyexp = []
        poly_symbols = self.predictable_polynomial_features(
            x=x_symbols,
            degree=degree
        )
        for xp in poly_symbols:
            polyexp.append([xp.count(s) for s in x_symbols])

        return polyexp

    def fit_native(self, X: List[List[NumericItem]], y: List[NumericItem], sample_weight=None):
        '''
        Wraps the standard fit method with native polynomial features
        Its main purpose is for native vs.predictable unit test comparisons,
        but you never know...

        :param X:
        :param y:
        :param sample_weight:
        :return:
        '''

        self.model = linear_model.LinearRegression(
            fit_intercept=(self.degree == 1))  # Or degree == 1 will kill everybody

        if self.degree == 1:
            self.model.fit(X, y, sample_weight)
        else:
            self.poly_def = preprocessing.PolynomialFeatures(
                degree=self.degree
            )
            X_poly = self.poly_def.fit_transform(X)
            self.model.fit(X_poly, y, sample_weight)

    def predict_native(self, x: List[NumericItem]):
        '''
        Wraps the standard predict method.
        Its main purpose is for native vs.predictable unit test comparisons,
        but you never know...

        :param x:
        :return:
        '''
        if self.degree == 1:
            pred = self.model.predict([x])
        else:
            x_poly = self.poly_def.fit_transform([x])
            pred = self.model.predict(x_poly)
        return pred

    def eval_polynomial_terms(self, polynomial_powers: List[List[int]], polynomial_coefficients: List[NumericItem],
                              x: List[NumericItem]):
        '''
        Eval the polynomial terms and return them separately

        :param polynomial_powers: Powers for each variable
        :param polynomial_coefficients: Coefficients
        :param x: Variable values (x is a vector)
        :return: List of monomials
        '''
        polynomial_terms = []
        for monomial_index, monomial_powers in enumerate(polynomial_powers):
            monomial = polynomial_coefficients[monomial_index]
            for power_index, power in enumerate(monomial_powers):
                monomial = monomial * (x[power_index] ** power)
            polynomial_terms.append(monomial)
        return polynomial_terms

    def create_polynomial_features(self, polynomial_powers: List[List[int]], x: List[NumericItem]):
        '''
        Create the numeric polynomial features (i.e. the x evaluated for all power
        combinations) given the powers and the values of x

        :param polynomial_powers:
        :param x:
        :return:
        '''
        polynomial_coefficients = [1] * len(polynomial_powers)
        return self.eval_polynomial_terms(polynomial_powers, polynomial_coefficients, x)

    def get_polynomial_definition(self):
        coefficients = \
            [self.model.intercept_[0]] + self.model.coef_[0].tolist() if self.fit_intercept else self.model.coef_[
                0].tolist()
        polynomial_definition = {
            LinearRegressionEngine.POLYNOMIAL_COEFFICIENTS_KEY: coefficients,
            LinearRegressionEngine.POLYNOMIAL_POWERS_KEY: self.polynomial_powers
        }
        return polynomial_definition

    def eval_polynomial_internal_do_not_use_me(self, polynomial_coefficients, x):
        '''
        Actually unused; TODO: Maybe Cleanup?

        :param polynomial_coefficients:
        :param x:
        :return:
        '''
        return sum(self.eval_polynomial_terms(polynomial_coefficients, x))

    def fit_predictable(self, X: List[List[NumericItem]], y: List[NumericItem], sample_weight=None):
        '''
        Fits data using polynomial regression and a predictable set of
        polynomial features.
        The latter, in order to allow for the definition of the polynomial
        to be exported and used regardless to the technology.

        :param X: Array of x multivariable samples
        :param y: Corresponding values
        :param sample_weight: Scipy LinearRegression corresponding parameter
        :return: None; sets

        '''
        self.number_of_variables = len(X[0])
        self.polynomial_powers = self.predictable_polynomial_powers(
            ["x" + str(x) for x in range(self.number_of_variables)],
            degree=self.degree
        )

        self.model = linear_model.LinearRegression(fit_intercept=self.fit_intercept)

        X_poly = []
        for xd1 in X:
            xi_poly = self.create_polynomial_features(self.polynomial_powers, xd1)
            X_poly.append(xi_poly)
        self.model.fit(X_poly, y, sample_weight)
