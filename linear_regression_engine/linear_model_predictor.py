class LinearModelPredictor:
    @classmethod
    def eval_polynomial(cls, coefficients, powers, x):
        '''

        :param coefficients:
        :param powers:
        :param x:
        :return:
        '''
        polynomial_terms = []
        for monomial_index, monomial_powers in enumerate(powers):
            monomial = coefficients[monomial_index]
            for power_index, power in enumerate(monomial_powers):
                monomial = monomial * (x[power_index] ** power)
            polynomial_terms.append(monomial)
        return sum(polynomial_terms), polynomial_terms
