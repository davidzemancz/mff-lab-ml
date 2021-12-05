# fbc0b6cc-0238-11eb-9574-ea7484399335
# 7b885094-03f8-11eb-9574-ea7484399335

import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

import smo_algorithm

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-7, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

def kernel(args: argparse.Namespace, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Use the kernel from the smo_algorithm assignment.
    return smo_algorithm.kernel(args, x, y)

def smo(
    args: argparse.Namespace,
    train_data: np.ndarray, train_target: np.ndarray,
    test_data: np.ndarray, test_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Use the SMO algorithm from the smo_algorithm assignment.
    return smo_algorithm.smo(args, train_data, train_target, test_data, test_target)

def main(args: argparse.Namespace) -> float:

    #x = np.array([1,2,3,1,2,3])
    #print(x)
    #print((x==2) | (x==3))
    #return 0

    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    results = np.zeros([test_data.shape[0], args.classes])

    # TODO: Using One-vs-One scheme, train (K \binom 2) classifiers, one for every
    # pair of classes $i < j$, using the `smo` method.
    for i in range(args.classes):
        for j in range(i + 1, args.classes):
            print("Training classes", i, "and", j)
            flags_IorJ_train = ((train_target == i) | (train_target == j))
            train_data_i_j = train_data[flags_IorJ_train]
            train_target_i_j = 2 * (train_target[flags_IorJ_train] == i) - 1

            flags_IorJ_test = ((test_target == i) | (test_target == j))
            test_data_i_j = test_data[flags_IorJ_test]
            test_target_i_j = 2 * (test_target[flags_IorJ_test] == i) - 1

            svs, svws, b, acc_train, acc_test = smo(args, train_data_i_j, train_target_i_j, test_data_i_j, test_target_i_j)

            sum = 0
            for sv, svw in zip(svs, svws):
               sum += svw * kernel(args, test_data, sv)
            predictions = b + sum
            
            results[:, i] += predictions > 0
            results[:, j] += predictions < 0

    # When training a classifier for classes $i < j$:
    # - keep only the training data of these classes, in the same order
    #   as in the input dataset;
    # - use targets 1 for the class $i$ and -1 for the class $j$.

    # Classify the test set by majority voting of all the trained classifiers,
    # using the lowest class index in the case of ties.
    #
    # Note that during prediction, only the support vectors returned by the `smo`
    # should be used, not all training data.
    #
    # Finally, compute the test set prediction accuracy.
    test_accuracy = sklearn.metrics.accuracy_score(test_target, np.argmax(results, axis=1))

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))
