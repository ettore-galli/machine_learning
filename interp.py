from linear_regression_engine.linear_regression_engine import LinearRegressionEngine
from linear_regression_engine.linear_model_predictor import LinearModelPredictor

if __name__ == "__main__":
    valori = [
        [7.0],
        [7.4],
        [7.9],
        [14.1],
        [17.2],
        [0.1]
    ]

    ingressi = [
        [200.0, 300.0, 50.0],
        [250.0, 200.0, 40.0],
        [350.0, 190.0, 55.0],
        [345.0, 180.0, 12.0],
        [371.0, 431.0, 77.0],
        [2.0, 3.0, 4.0]
    ]

    tests_base = [
        [130.0, 200.0, 10.0],
        [260.0, 201.0, 31.0],
        [590.0, 320.0, 98.0],
        [371.0, 431.0, 77.0],
        [371.1, 431.1, 77.1]
    ]

    # /Users/ettoregalli/Documents/SVILUPPO/python/venv3/bin/python

    DEGREE = 2

    print("========== PREDITTORE NATIVO")

    lre = LinearRegressionEngine(degree=DEGREE)
    lre.fit(ingressi, valori)

    if DEGREE > 1:
        print(
            "\nself.poly_def.get_feature_names()",
            lre.poly_def.get_feature_names()
        )
    print(
        "\ncoef",
        # lre.model.intercept_[0],
        lre.model.coef_[0]

    )

    print(lre.predict_native(tests_base[4]))
    print(lre.predict_native([2, 3, 4]))

    print("========== PREDITTORE PREDICIBILE")
    lre2 = LinearRegressionEngine(degree=DEGREE)
    lre2.fit_predictable(ingressi, valori)

    print("----------------")
    ppf = lre2.predictable_polynomial_features(['a', 'b', 'c'], DEGREE)
    print(ppf)
    poly = lre2.get_polynomial_definition()
    print(poly)

    pred = LinearModelPredictor.eval_polynomial(poly["coefficients"], poly["powers"], tests_base[4])
    print(pred)
    pred = LinearModelPredictor.eval_polynomial(poly["coefficients"], poly["powers"], [2, 3, 4])
    print(pred)

    '''
    self.poly_def.get_feature_names() 
    f = ['1', 'x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2']


    c = [0.14003618930935247, -3.89106607e-16,  4.66854152e-03,  4.90840477e-03,  7.85460918e-04, -1.68921721e-05,  7.84762917e-05,  1.07058613e-03, 2.49063725e-04, -1.56055102e-03, -3.73270436e-03]

    
    
    '''