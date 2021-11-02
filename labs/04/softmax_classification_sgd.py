# fbc0b6cc-0238-11eb-9574-ea7484399335
# 7b885094-03f8-11eb-9574-ea7484399335

import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def softmax(x):
    """
    Computes softmax of x (np array)
    It's normalized (x - np.max(x))
    """
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x - np.max(x)))
    return f_x

def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    
    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Train target to one hot representation
    train_target_oh = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore").fit_transform(np.reshape(train_target, (-1,1)))

    # Generate initial model weights as matrix count_of_features * count_of_classes
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)
        
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.
        batches_count = train_data.shape[0] / args.batch_size
        for batch in range(1, int(batches_count) + 1):
            gradient_sum = np.zeros([train_data.shape[1]])
            for i in range(args.batch_size * (batch - 1), args.batch_size * batch):
                p = permutation[i]
                x_i = train_data[p]
                t_i = train_target_oh[p] # One-hot represenatation of target data (same size as y-i)
                y_i = softmax(x_i.transpose() @ weights) # Vector of size args.classes (sigmoid for each class) ... probrabilities of each class

                # Convert to 2D vectors
                x_i = np.reshape(x_i, (-1,1))
                t_i = np.reshape(t_i, (-1,1))
                y_i = np.reshape(y_i, (-1,1))

                gradient = (y_i - t_i) @ x_i.T # Gradient is matrix
                gradient_sum = gradient_sum + gradient
            
            gradient_avrg = gradient_sum / args.batch_size
            weights = weights - (args.learning_rate * gradient_avrg).T

        # After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.
        train_accuracy, train_loss, test_accuracy, test_loss = 0, 0, 0, 0
        
        train_predictions = np.zeros([train_target.shape[0], args.classes])
        i = 0
        for x_i in train_data:
           train_predictions[i] = softmax((x_i.transpose() @ weights))
           i = i + 1
        # Compute loss and accuracy (using train_target as NON-one-hot and train_predictions as list of vectors of probabilities)
        train_loss = sklearn.metrics.log_loss(train_target, train_predictions)
        train_accuracy = sklearn.metrics.accuracy_score(train_target, np.argmax(train_predictions, axis=1))

        test_predictions = np.zeros([test_target.shape[0], args.classes])
        i = 0
        for x_i in test_data:
           test_predictions[i] = softmax((x_i.transpose() @ weights))
           i = i + 1
        # Compute loss and accuracy
        test_loss = sklearn.metrics.log_loss(test_target, test_predictions)
        test_accuracy = sklearn.metrics.accuracy_score(test_target, np.argmax(test_predictions, axis=1))

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, train_accuracy), (test_loss, test_accuracy)]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:", *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
