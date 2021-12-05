# fbc0b6cc-0238-11eb-9574-ea7484399335
# 7b885094-03f8-11eb-9574-ea7484399335

import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-7, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

def kernel(args: argparse.Namespace, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    if args.kernel == "poly": return (args.kernel_gamma * np.dot(x, z) + 1) ** args.kernel_degree
    elif args.kernel == "rbf": return np.exp(-args.kernel_gamma * np.sum((x - z) ** 2, axis=-1))

# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(
    args: argparse.Namespace,
    train_data: np.ndarray, train_target: np.ndarray,
    test_data: np.ndarray, test_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    train_data_length = train_data.shape[0]
    test_data_length = test_data.shape[0]

    # Prepare kernel matrices
    K_train = np.zeros([train_data_length, train_data_length])
    K_test = np.zeros([test_data_length, train_data_length])

    # Precompute kernel matrix for train data
    K_train = np.array([[kernel(args, x1, z1) for x1 in train_data] for z1 in train_data])
          
    # Precompute kernel matrix for test data
    K_test = np.array([[kernel(args, x1, z1) for x1 in test_data] for z1 in train_data])

    passes_without_as_changing = 0
    train_accs, test_accs = [], []
    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i
            j = j + (j >= i)

            #  Check that a[i] fulfils the KKT conditions, using `args.tolerance` during comparisons.
            y_xi = ((a * train_target) @ K_train[i]) + b
            E_i = y_xi - train_target[i]
            if not (a[i] < args.C - args.tolerance and (train_target[i] * E_i) < -args.tolerance) and not (a[i] > args.tolerance and (train_target[i] * E_i) > args.tolerance):
                continue
            else:
                if 2 * K_train[i, j] - K_train[i, i] - K_train[j, j] > -args.tolerance: 
                    continue
                else:
                    # Compute new aj
                    y_xj = (a * train_target) @ K_train[j] + b
                    E_j = y_xj - train_target[j]
                    aj_new = a[j] - train_target[j] * (E_i - E_j) / (2 * K_train[i, j] - K_train[i, i] - K_train[j, j])

                    if train_target[i] != train_target[j]:
                        L = max(0, a[j] - a[i]),
                        H = min(args.C,  args.C - a[i] + a[j])
                    else:
                        L = max(0, a[i] + a[j] - args.C),
                        H = min(args.C, a[i] + a[j])

                    # Compute new aj and check
                    aj_new = np.clip(aj_new, L, H)
                    if abs(aj_new - a[j]) < args.tolerance: 
                        continue
                    else:
                        # Compute new ai
                        ai_new = a[i] - train_target[i] * train_target[j] * (aj_new - a[j])

                        # Compute new bi, bj
                        bi_new = b - E_i - train_target[i] * (ai_new - a[i]) * K_train[i, i] - train_target[j] * (aj_new - a[j]) * K_train[j, i]
                        bj_new = b - E_j - train_target[i] * (ai_new - a[i]) * K_train[i, j] - train_target[j] * (aj_new - a[j]) * K_train[j, j]
                        a[i] = ai_new
                        a[j] = aj_new

                        # Check tolerances and set b
                        if args.tolerance < a[i] and a[i] < args.C - args.tolerance:
                            b = bi_new
                        elif args.tolerance < a[j] and a[j] < args.C - args.tolerance:
                            b = bj_new
                        else:
                            b = (bi_new + bj_new) / 2

                        # Changed
                        as_changed = as_changed + 1

        # After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.
        preds_train = a * train_target @ K_train + b
        acc_train = sklearn.metrics.accuracy_score(train_target, np.sign(preds_train))
        train_accs.append(acc_train)

        preds_test = a * train_target @ K_test + b
        acc_test = sklearn.metrics.accuracy_score(test_target, np.sign(preds_test))
        test_accs.append(acc_test)

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
    support_vectors = train_data[a > args.tolerance]

    support_vector_weights = (a * train_target)[a > args.tolerance]

    print("Done, iteration {}, support vectors {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), len(support_vectors), 100 * train_accs[-1], 100 * test_accs[-1]))

    return support_vectors, support_vector_weights, b[0], train_accs, test_accs

def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt
        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#00dd00")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ffff00")
            plt.legend(loc="upper center", ncol=4)

        # If you want plotting to work (not required for ReCodEx), you need to
        # define `predict_function` computing SVM value `y(x)` for the given x.
        predict_function = lambda x: None

        plot(predict_function, support_vectors)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
