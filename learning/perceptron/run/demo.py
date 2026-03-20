from perceptron.polynomial_features import numerical_polynomial_features


if __name__ == "__main__":
    data = [1.25, 2.35]
    got = numerical_polynomial_features(data, degree=3)
    print(got)
