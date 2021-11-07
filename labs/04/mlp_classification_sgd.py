#!/usr/bin/env python3
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
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--iterations", default=10, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> tuple[tuple[np.ndarray, ...], list[float]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    # -------------
    #   Useful link:
    #       https://rstudio-pubs-static.s3.amazonaws.com/337306_79a7966fad184532ab3ad66b322fe96e.html
    # -------------

    def softmax(x):
        """
        Computes softmax of x (np array)
        It's normalized (x - np.max(x))
        """
        y = np.exp(x - np.max(x))
        f_x = y / np.sum(np.exp(x - np.max(x)))
        return f_x
        
    def ReLU(x):
        """
        Returns max(x, 0)
        """
        return max(x, 0)

    def forward(inputs):
        # Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where ReLU(x) = max(x, 0), and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as ReLU(inputs @ weights[0] + biases[0]).
        # The value of the output layer is computed as softmax(hidden_layer @ weights[1] + biases[1]).
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.
        
        hLayer_output = ReLU(inputs @ weights[0] + biases[0])
        oLayer_output = softmax(hLayer_output @ weights[1] + biases[1])
        return (hLayer_output, oLayer_output)


    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        # Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        batches_count = train_data.shape[0] / args.batch_size
        for batch in range(1, int(batches_count) + 1):

            # Gradient
            gradient_weights_sum = np.zeros(weights.shape)
            gradient_biases_sum = np.zeros(biases.shape)
            
            # For each sample
            for i in range(args.batch_size * (batch - 1), args.batch_size * batch):
                p = permutation[i]
                x_i = train_data[p]
                t_i = train_target[p]
                activation_result = forward(x_i)
            
            # Update weights (both, on hidden and output layer)
            gradient_weights_avrg = gradient_weights_sum / args.batch_size
            weights = weights - (args.learning_rate * gradient_weights_avrg)

            # Update biases (both, on hidden and output layer)
            gradient_biases_avrg = gradient_biases_sum / args.batch_size
            biases = biases - (args.learning_rate * gradient_biases_avrg)


        # The gradient used in SGD has now four parts, gradient of weights[0] and weights[1]
        # and gradient of biases[0] and biases[1].
        #
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of -log P(target | data), or you can compute
        # it step by step using the chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer
        # - compute the derivative with respect to weights[1] and biases[1]
        # - compute the derivative with respect to the hidden layer output
        # - compute the derivative with respect to the hidden layer input
        # - compute the derivative with respect to weights[0] and biases[0]

        # TODO: After the SGD iteration, measure the accuracy for both the
        # train test and the test set.
        train_accuracy, test_accuracy = None, None

        print("After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
            iteration + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases), [train_accuracy, test_accuracy]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(args)
    print("Learned parameters:", *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:20]] + ["..."]) for ws in parameters), sep="\n")
