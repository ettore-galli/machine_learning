# pylint: disable=too-many-lines
import numpy as np


from perceptron.perceptron_algorithm import (
    origin_perceptron_step,
    perceptron,
    Data,
    Labels,
    Params,
    Sample,
    Label,
    Theta,
    ThetaZero,
)


def log_classifier_step(sample: Sample, label: Label, theta: Theta, theta_0=ThetaZero):
    print(sample, label, theta, theta_0)


def test_hw_1a1b():
    data: Data = np.array([[200, 800, 200, 800], [0.2, 0.2, 0.8, 0.8], [1, 1, 1, 1]])
    labels: Labels = np.array([[-1, -1, 1, 1]])
    theta = np.array(
        [
            [0],
            [1],
            [-0.5],
        ]
    )
    print("\n")
    print("data", data, data.shape)
    print("labels", labels, labels.shape)
    print("theta", theta, theta.shape)

    margins = labels * np.dot(theta.T, data) / np.linalg.norm(theta)

    margin = margins.min()
    print("\n")

    print("margins, margin")
    print(margins, margin)

    radius = max(np.linalg.norm(data, axis=0))
    errors = (radius / margin) ** 2

    print("errors")
    print(errors)


def test_hw_1c():
    data: Data = np.array([[200, 800, 200, 800], [0.2, 0.2, 0.8, 0.8], [1, 1, 1, 1]])
    labels: Labels = np.array([[-1, -1, 1, 1]])
    params: Params = {"T": 1000}

    classifier = perceptron(
        data=data,
        labels=labels,
        params=params,
        perceptron_step=origin_perceptron_step,
        hook=None,
    )

    classifier_coefficents = classifier.get_classifier_coefficents()
    print(classifier)
    print(classifier_coefficents)
