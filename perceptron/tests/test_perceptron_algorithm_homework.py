# pylint: disable=too-many-lines
import numpy as np


from perceptron.perceptron_algorithm import (
    offset_perceptron_step,
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
    # data: Data = np.array([[200, 800, 200, 800], [0.2, 0.2, 0.8, 0.8], [1, 1, 1, 1]])
    data: Data = np.array([[0.2, 0.8, 0.2, 0.8], [0.2, 0.2, 0.8, 0.8], [1, 1, 1, 1]])
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


def test_hw_1d():
    # data: Data = np.array(
    #     [[0.2, 0.8, 0.2, 0.8], [0.0002, 0.0002, 0.0008, 0.0008], [1, 1, 1, 1]]
    # )
    data: Data = np.array([[0.2, 0.8, 0.2, 0.8], [0.2, 0.2, 0.8, 0.8], [1, 1, 1, 1]])
    labels: Labels = np.array([[-1, -1, 1, 1]])
    # theta = np.array(
    #     [
    #         [0],
    #         [1],
    #         [-0.0005],
    #     ]
    # )
    theta = np.array(
        [
            [0],
            [1],
            [-0.5],
        ]
    )
    print("\n")
    print("data:\n", data, data.shape)
    print("labels:\n", labels, labels.shape)
    print("theta:\n", theta, theta.shape)

    margins = labels * np.dot(theta.T, data) / np.linalg.norm(theta)

    margin = margins.min()
    print("\n")

    print("margins, margin")
    print(margins, margin)

    radius = max(np.linalg.norm(data, axis=0))
    errors = (radius / margin) ** 2

    print("errors")
    print(errors)


# pylint: disable=duplicate-code
def test_hw_2a():
    # data: Data = np.array([[200, 800, 200, 800], [0.2, 0.2, 0.8, 0.8], [1, 1, 1, 1]])
    data: Data = np.array([[2, 3, 4, 5]])
    labels: Labels = np.array([[1, 1, -1, -1]])
    params: Params = {"T": 1000}

    classifier = perceptron(
        data=data,
        labels=labels,
        params=params,
        perceptron_step=offset_perceptron_step,
        hook=None,
    )

    classifier_coefficents = classifier.get_classifier_coefficents()
    print(classifier)
    print(classifier_coefficents)


# pylint: disable=duplicate-code
def test_hw_2e():
    # 2 3 4 5
    data: Data = np.array(
        [
            [0, 0, 0, 0],  # 1 Samsung
            [1, 0, 0, 0],  # 2 Xiaomi
            [0, 1, 0, 0],  # 3 Sony
            [0, 0, 1, 0],  # 4 Apple
            [0, 0, 0, 1],  # 5 LG
            [0, 0, 0, 0],  # 6 Nokia
        ]
    )

    labels: Labels = np.array([[1, 1, -1, -1]])
    params: Params = {"T": 1000}

    classifier = perceptron(
        data=data,
        labels=labels,
        params=params,
        perceptron_step=offset_perceptron_step,
        hook=None,
    )

    classifier_coefficents = classifier.get_classifier_coefficents()
    print(classifier)
    print(classifier_coefficents)
    theta = classifier_coefficents[0]
    my_data: Data = np.array(
        [
            [1, 0],  # 1 Samsung
            [0, 0],  # 2 Xiaomi
            [0, 0],  # 3 Sony
            [0, 0],  # 4 Apple
            [0, 0],  # 5 LG
            [0, 1],  # 6 Nokia
        ]
    )
    my_labels = np.array([-1, -1])

    margins = my_labels * np.dot(theta.T, my_data) / np.linalg.norm(theta)
    print(margins)


# pylint: disable=duplicate-code
def test_hw_2g():
    # 2 3 4 5
    data: Data = np.array(
        [
            [1, 0, 0, 0, 0, 0],  # 1 Samsung
            [0, 1, 0, 0, 0, 0],  # 2 Xiaomi
            [0, 0, 1, 0, 0, 0],  # 3 Sony
            [0, 0, 0, 1, 0, 0],  # 4 Apple
            [0, 0, 0, 0, 1, 0],  # 5 LG
            [0, 0, 0, 0, 0, 1],  # 6 Nokia
        ]
    )

    labels: Labels = np.array([[1, 1, -1, -1, 1, 1]])
    params: Params = {"T": 1000000000000}

    classifier = perceptron(
        data=data,
        labels=labels,
        params=params,
        perceptron_step=offset_perceptron_step,
        hook=None,
    )

    classifier_coefficents = classifier.get_classifier_coefficents()
    print(classifier)
    print(classifier_coefficents)
