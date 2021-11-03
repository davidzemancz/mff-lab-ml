# fbc0b6cc-0238-11eb-9574-ea7484399335
# 7b885094-03f8-11eb-9574-ea7484399335

import argparse
import sys
import math

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

# Sigmoid function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    data, target = sklearn.datasets.make_classification(n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)

    # Append a constant feature with value 1 to the end of every input data
    data = np.append(data, np.ones([1,data.shape[0]]).transpose(), axis=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial linear regression weights
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        # Same as linear_regression_sgd.py in 02
        batches_count = train_data.shape[0] / args.batch_size
        for batch in range(1, int(batches_count) + 1):
            gradient_sum = np.zeros([train_data.shape[1]])
            for i in range(args.batch_size * (batch - 1), args.batch_size * batch):
                p = permutation[i]
                x_i = train_data[p]
                t_i = train_target[p]
                gradient = (sigmoid((x_i.transpose() @ weights)) - t_i) * x_i
                gradient_sum = gradient_sum + gradient
            
            gradient_avrg = gradient_sum / args.batch_size
            weights = weights - (args.learning_rate * gradient_avrg)

        # After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.
        train_accuracy, train_loss, test_accuracy, test_loss = 0, 0, 0, 0
        
        train_predictions = np.zeros(train_target.shape)
        i = 0
        for x_i in train_data:
           train_predictions[i] = sigmoid((x_i.transpose() @ weights))
           i = i + 1
        train_loss = sklearn.metrics.log_loss(train_target, train_predictions)
        train_accuracy = sklearn.metrics.accuracy_score(train_target, sklearn.preprocessing.binarize(train_predictions.reshape(-1, 1), threshold=0.5))

        test_predictions = np.zeros(test_target.shape)
        i = 0
        for x_i in test_data:
           test_predictions[i] = sigmoid((x_i.transpose() @ weights))
           i = i + 1
        test_loss = sklearn.metrics.log_loss(test_target, test_predictions)
        test_accuracy = sklearn.metrics.accuracy_score(test_target, sklearn.preprocessing.binarize(test_predictions.reshape(-1, 1), threshold=0.5))

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                if not epoch: plt.figure(figsize=(6.4*3, 4.8*(args.epochs+2)//3))
                plt.subplot(3, (args.epochs+2)//3, 1 + epoch)
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=21, cmap=plt.cm.RdBu, alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="P", label="train", cmap=plt.cm.RdBu)
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, label="test", cmap=plt.cm.RdBu)
            plt.legend(loc="upper right")
            if args.plot is True: plt.show()
            else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, [(train_loss, train_accuracy), (test_loss, test_accuracy)]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))
