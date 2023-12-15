from __future__ import annotations
from dataclasses import dataclass
from functools import reduce


from typing import Any, Callable, Dict, Protocol, Tuple, Optional, Generator
import numpy as np

from perceptron.iteration import iterate_while

PERCEPTRON_DEFAULT_ITERATIONS: int = 100

Data = np.ndarray
Labels = np.ndarray

Sample = np.ndarray
Label = float

Params = Dict[str, Any]

Theta = np.ndarray
ThetaZero = float


Hook = Callable[[Sample, Label, Theta, ThetaZero], None]


# pylint: disable=too-many-instance-attributes
@dataclass
class Classifier:
    is_averaged: bool
    dimension: int
    theta: Theta
    theta_0: ThetaZero
    theta_sum: Theta
    theta_0_sum: ThetaZero
    number_of_runs: int
    has_mistakes: bool = False

    @staticmethod
    def initial(dimension: int, is_averaged: bool) -> Classifier:
        return Classifier(
            is_averaged=is_averaged,
            dimension=dimension,
            theta=np.zeros(dimension),
            theta_0=0,
            theta_sum=np.zeros(dimension),
            theta_0_sum=0,
            number_of_runs=0,
            has_mistakes=False,
        )

    def with_mistake_correction(
        self, delta_theta: Theta, delta_theta_0: ThetaZero
    ) -> Classifier:
        theta = self.theta + delta_theta
        theta_0 = self.theta_0 + delta_theta_0

        return Classifier(
            is_averaged=self.is_averaged,
            dimension=self.dimension,
            theta=theta,
            theta_0=theta_0,
            theta_sum=self.theta_sum,
            theta_0_sum=self.theta_0_sum,
            has_mistakes=True,
            number_of_runs=self.number_of_runs,
        )

    def with_average_data(self) -> Classifier:
        return Classifier(
            is_averaged=self.is_averaged,
            dimension=self.dimension,
            theta=self.theta,
            theta_0=self.theta_0,
            theta_sum=self.theta_sum + self.theta,
            theta_0_sum=self.theta_0_sum + self.theta_0,
            has_mistakes=self.has_mistakes,
            number_of_runs=self.number_of_runs + 1,
        )

    def get_classifier_coefficents(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            (
                self.theta_avg.reshape((self.dimension, 1)),
                np.array([self.theta_0_avg]),
            )
            if self.is_averaged
            else (
                self.theta.reshape((self.dimension, 1)),
                np.array([self.theta_0]),
            )
        )

    @property
    def theta_avg(self) -> Theta:
        return (
            self.theta_sum / self.number_of_runs
            if self.number_of_runs > 0
            else self.theta_sum
        )

    @property
    def theta_0_avg(self) -> ThetaZero:
        return (
            self.theta_0_sum / self.number_of_runs
            if self.number_of_runs > 0
            else self.theta_0_sum
        )


# pylint: disable=too-few-public-methods
class PerceptronStepProtocol(Protocol):
    def __call__(
        self,
        classifier: Classifier,
        sample: Sample,
        label: Label,
        hook: Optional[Hook] = None,
    ) -> Classifier:
        ...


class PerceptronAlgorithmProtocol(Protocol):
    # pylint: disable=too-many-arguments
    def __call__(
        self,
        data: Data,
        labels: Labels,
        params: Params,
        perceptron_step: PerceptronStepProtocol,
        hook: Optional[Hook] = None,
    ) -> Classifier:
        ...


def offset_perceptron_step(
    classifier: Classifier, sample: Sample, label: Label, hook: Optional[Hook] = None
) -> Classifier:
    result = np.dot(classifier.theta, sample) + classifier.theta_0
    margin = label * result

    if margin <= 0:
        classifier_with_mistake_correction = classifier.with_mistake_correction(
            delta_theta=label * sample, delta_theta_0=label
        )
        if hook:
            hook(
                sample,
                label,
                classifier_with_mistake_correction.theta,
                classifier_with_mistake_correction.theta_0,
            )

        return classifier_with_mistake_correction.with_average_data()

    return classifier.with_average_data()


# pylint: disable=too-many-arguments
def perceptron_engine(
    data: Data,
    labels: Labels,
    params: Params,
    is_averaged: bool,
    perceptron_step: PerceptronStepProtocol,
    hook: Optional[Hook] = None,
) -> Classifier:
    dimension = data.shape[0]

    classifier = Classifier.initial(dimension=dimension, is_averaged=is_averaged)

    def single_sample_reducer(acc: Classifier, cur: Tuple[Sample, Label]):
        return perceptron_step(classifier=acc, sample=cur[0], label=cur[1], hook=hook)

    def perceptron_iteration_function(cur_classifier: Classifier) -> Classifier:
        return reduce(single_sample_reducer, zip(data.T, labels.T), cur_classifier)

    def perceptron_iteration_predicate(cur_classifier: Classifier) -> bool:
        return cur_classifier.has_mistakes

    return iterate_while(
        initial=classifier,
        iteration_function=perceptron_iteration_function,
        while_predicate=perceptron_iteration_predicate,
        maximum_iterations=params.get("T", PERCEPTRON_DEFAULT_ITERATIONS),
        evaluate_predicate_post=True,
    )


def perceptron(
    data: Data,
    labels: Labels,
    params: Params,
    perceptron_step: PerceptronStepProtocol,
    hook: Optional[Hook] = None,
) -> Classifier:
    return perceptron_engine(
        data=data,
        labels=labels,
        params=params,
        is_averaged=False,
        perceptron_step=perceptron_step,
        hook=hook,
    )


def averaged_perceptron(
    data: Data,
    labels: Labels,
    params: Params,
    perceptron_step: PerceptronStepProtocol,
    hook: Optional[Hook] = None,
) -> Classifier:
    return perceptron_engine(
        data=data,
        labels=labels,
        params=params,
        is_averaged=True,
        perceptron_step=perceptron_step,
        hook=hook,
    )


def averaged_perceptron_legacy(
    data: Data,
    labels: Labels,
    params: Params,
    perceptron_step: PerceptronStepProtocol,
    hook: Optional[Hook] = None,
) -> Classifier:
    dimension = data.shape[0]
    theta = np.zeros(dimension)
    theta_0 = 0
    theta_avg = np.zeros(dimension)
    theta_0_avg = 0
    number_of_runs = 0
    _ = perceptron_step
    for _ in range(params.get("T", 10)):
        for sample, label in zip(data.T, labels.T):
            result = np.dot(theta, sample) + theta_0
            margin = label * result

            if margin <= 0:
                theta += label * sample
                theta_0 += label
                if hook:
                    hook(sample, label, theta, theta_0)
            theta_avg += theta
            theta_0_avg += theta_0
            number_of_runs += 1

    return Classifier(
        is_averaged=True,
        dimension=dimension,
        theta=theta,
        theta_0=theta_0,
        theta_sum=theta_avg,
        theta_0_sum=theta_0_avg,
        number_of_runs=number_of_runs,
    )


def y(x, th, th0):
    return np.dot(np.transpose(th), x) + th0


def positive(x, th, th0):
    return np.sign(y(x, th, th0))


def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels)


def eval_classifier(
    learner: PerceptronAlgorithmProtocol,
    data_train,
    labels_train,
    data_test: np.ndarray,
    labels_test,
):
    theta, theta_0 = learner(
        data=data_train,
        labels=labels_train,
        params={"T": 100},
        perceptron_step=offset_perceptron_step,
        hook=None,
    ).get_classifier_coefficents()

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
