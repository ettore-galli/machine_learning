from __future__ import annotations
from dataclasses import dataclass
from functools import reduce


from typing import Any, Callable, Dict, Tuple, Optional, Generator
import numpy as np

from perceptron.iteration import reduce_until

PERCEPTRON_DEFAULT_ITERATIONS: int = 100

Data = np.ndarray
Labels = np.ndarray

Sample = np.ndarray
Label = float

Params = Dict[str, Any]

Theta = np.ndarray
ThetaZero = float

Hook = Callable[[Theta, ThetaZero], None]


@dataclass
class Classifier:
    theta: Theta
    theta_0: ThetaZero
    has_mistakes: bool = False

    def with_mistake_correction(
        self, delta_theta: Theta, delta_theta_0: ThetaZero
    ) -> Classifier:
        return Classifier(
            theta=self.theta + delta_theta,
            theta_0=self.theta_0 + delta_theta_0,
            has_mistakes=True,
        )


def perceptron_step(
    classifier: Classifier, sample: Sample, label: Label, hook: Optional[Hook] = None
) -> Classifier:
    result = np.dot(classifier.theta, sample) + classifier.theta_0
    margin = label * result

    if margin <= 0:
        classifier_with_mistake_correction = classifier.with_mistake_correction(
            delta_theta=label * sample, delta_theta_0=label
        )
        if hook:
            hook(classifier.theta, classifier.theta_0)
        return classifier_with_mistake_correction

    return classifier


def perceptron(
    data: Data, labels: Labels, params: Params, hook: Optional[Hook] = None
) -> Tuple[np.ndarray, np.ndarray]:
    dimension = data.shape[0]

    classifier = Classifier(theta=np.zeros(dimension), theta_0=0)

    def single_sample_reducer(acc, cur):
        return perceptron_step(classifier=acc, sample=cur[0], label=cur[1], hook=hook)

    def dataset_iterations_reducer(acc_classifier, _):
        return reduce(single_sample_reducer, zip(data.T, labels.T), acc_classifier)

    classifier = reduce_until(
        function=dataset_iterations_reducer,
        sequence=range(params.get("T", PERCEPTRON_DEFAULT_ITERATIONS)),
        initial=classifier,
        predicate=lambda acc, _: not acc.has_mistakes,
    )

    return classifier.theta.reshape((dimension, 1)), np.array([classifier.theta_0])


def averaged_perceptron(
    data: Data, labels: Labels, params: Params, hook: Optional[Hook] = None
) -> Tuple[np.ndarray, np.ndarray]:
    dimension = data.shape[0]
    theta = np.zeros(dimension)
    theta_0 = 0
    theta_avg = np.zeros(dimension)
    theta_0_avg = 0
    number_of_runs = 0
    for _ in range(params.get("T", 10)):
        for sample, label in zip(data.T, labels.T):
            result = np.dot(theta, sample) + theta_0
            margin = label * result

            if margin <= 0:
                theta += label * sample
                theta_0 += label
                if hook:
                    hook(theta, theta_0)
            theta_avg += theta
            theta_0_avg += theta_0
            number_of_runs += 1

    return (theta_avg / number_of_runs).reshape((dimension, 1)), np.array(
        [theta_0_avg / number_of_runs]
    )


def y(x, th, th0):
    return np.dot(np.transpose(th), x) + th0


def positive(x, th, th0):
    return np.sign(y(x, th, th0))


def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels)


def eval_classifier(
    learner, data_train, labels_train, data_test: np.ndarray, labels_test
):
    theta, theta_0 = learner(
        data=data_train, labels=labels_train, params={"T": 100}, hook=None
    )

    return score(data_test, labels_test, theta, theta_0) / data_test.shape[1]


def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    evaluations = []

    for _ in range(it):
        data_train, labels_train = data_gen(n_train)
        data_test, labels_test = data_gen(n_test)
        evaluations.append(
            eval_classifier(
                learner=learner,
                data_train=data_train,
                labels_train=labels_train,
                data_test=data_test,
                labels_test=labels_test,
            )
        )
    return sum(evaluations) / len(evaluations) if len(evaluations) > 0 else 0


def eval_learning_alg_same(learner, data_gen, n_data, it):
    evaluations = []

    for _ in range(it):
        data, labels = data_gen(n_data)

        evaluations.append(
            eval_classifier(
                learner=learner,
                data_train=data,
                labels_train=labels,
                data_test=data,
                labels_test=labels,
            )
        )
    return sum(evaluations) / len(evaluations) if len(evaluations) > 0 else 0


def d_split_j(data: np.ndarray, k: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
    length = data.shape[1]
    if length % k == 0:
        part_length = length // k
        part_start = j * part_length
        part_end = (j + 1) * part_length
    else:
        part_length = length // k + 1
        part_start = j * part_length
        part_end = (j + 1) * part_length

    return data[:, part_start:part_end], np.concatenate(
        (data[:, :part_start], data[:, part_end:]), axis=1
    )


def d_split_j_looper(
    data: np.ndarray, labels: np.ndarray, k: int
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    for j in range(k):
        yield d_split_j(data, k, j) + d_split_j(labels, k, j)


def xval_learning_alg(learner, data, labels, k):
    # cross validation of learning algorithm

    scores = []

    data_parts = np.array_split(data, k, axis=1)
    labels_parts = np.array_split(labels, k, axis=1)

    for j in range(k):
        data_test = data_parts[j]
        data_train = np.concatenate(
            [d for i, d in enumerate(data_parts) if i != j], axis=1
        )
        labels_test = labels_parts[j]
        labels_train = np.concatenate(
            [d for i, d in enumerate(labels_parts) if i != j], axis=1
        )

        scores.append(
            eval_classifier(learner, data_train, labels_train, data_test, labels_test)
        )
    return sum(scores) / len(scores)
