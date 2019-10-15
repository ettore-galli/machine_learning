import numpy as np

from sklearn import linear_model
if __name__ == "__main__":

    tests_base = [
        [130.0, 200.0, 10.0],
        [260.0, 201.0, 31.0],
        [590.0, 320.0, 98.0],
        [371.0, 431.0, 77.0],
        [371.1, 431.1, 77.1]
    ]

    print("Carico il predittore")

    import pickle

    fqn = "./data/modello.save"
    f = open(fqn, mode="rb")
    rsave = f.read()
    r = pickle.loads(rsave)

    print("Uso il predittore")

    try:
        for test in tests_base:
            pred = r.predict([test])
            print(test, pred)
    except Exception as e:
        print(e)
