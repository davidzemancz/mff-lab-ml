# fbc0b6cc-0238-11eb-9574-ea7484399335
# 7b885094-03f8-11eb-9574-ea7484399335

import argparse

import numpy as np
import sklearn.metrics

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--iterations", default=200, type=int, help="Number of training iterations")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.

     
def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1

    test_data = np.linspace(-1.2, 1.2, 2 * args.data_size)
    test_target = np.sin(5 * test_data) + 1
    
    # TODO: Perform `args.iterations` of SGD-like updates, but in dual formulation
    # using `betas` as weights of individual training examples.
    #
    # We assume the primary formulation of our model is
    #   y = phi(x)^T w + bias
    # and the loss in the primary problem is batched MSE with L2 regularization:
    #   L = sum_{i \in B} 1/|B| * [1/2 * (phi(x_i)^T w + bias - target_i)^2] + 1/2 * args.l2 * w^2
    # Regarding the L2 regularization, note that it always affects all betas, not
    # just the ones in the batch.
    #
    # For `bias`, use explicitly the average of the training targets, and do
    # not update it futher during training.
    #
    # Instead of using feature map `phi` directly, we use a given kernel computing
    #   K(x, y) = phi(x)^T phi(y)
    # We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    def kernel (x, z):
        if args.kernel == "poly": return (args.kernel_gamma * np.dot(x, z) + 1) ** args.kernel_degree
        elif args.kernel == "rbf": return np.exp(-args.kernel_gamma * ((x - z) ** 2))
  
    
    # Betas
    betas = np.zeros(args.data_size)

    # Bias
    bias = np.mean(train_target)

    # Count of batches
    batches = train_data.shape[0] / args.batch_size

    # Rmses    
    train_rmses, test_rmses = [], []

    K_train = np.zeros([args.data_size, args.data_size])
    K_test = np.zeros([args.data_size * 2, args.data_size])

    # Precompute kernel matrix for train data
    for i in range(args.data_size):
        for j in range(args.data_size):
            K_train[i, j] = kernel(train_data[i], train_data[j])

    # Precompute kernel matrix for test data
    for i in range(args.data_size * 2):
        for j in range(args.data_size):
            K_test[i, j] = kernel(test_data[i], train_data[j])

      # After each iteration, compute RMSE both on training and testing data.
    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])
        permutation_len = len(permutation)
        
        # Process the data in the order of `permutation`, performing
        # batched updates to the `betas`. You can assume that `args.batch_size`
        # exactly divides `train_data.shape[0]`.
        for batch_start in range(0, permutation_len, args.batch_size):
            batch_interval = permutation[batch_start : (batch_start + args.batch_size)]
            
            # Update betas in batch
            betas[batch_interval] = betas[batch_interval] - ((args.learning_rate * (((K_train[batch_interval] @ betas) + bias) - train_target[batch_interval])) / args.batch_size)

            # L2 reg for all betas
            betas = betas - (args.learning_rate * args.l2 * betas)
        
        # Append RMSE on training and testing data to `train_rmses` and
        # `test_rmses` after the iteration.

        #rint(betas.shape)
        #print(K_train.shape)
        #print(K_test.shape)

        # Create predictions
        train_predictions = (K_train @ betas) + bias
        test_predictions = (K_test @ betas) + bias

        train_rmses.append(sklearn.metrics.mean_squared_error(train_target, train_predictions, squared=False))
        test_rmses.append(sklearn.metrics.mean_squared_error(test_target, test_predictions, squared=False))

        if (iteration + 1) % 10 == 0:
            print("Iteration {}, train RMSE {:.2f}, test RMSE {:.2f}".format(
                iteration + 1, train_rmses[-1], test_rmses[-1])) #test_rmses[-1]

    if args.plot:
        import matplotlib.pyplot as plt
        # If you want the plotting to work (not required for ReCodEx), compute the `test_predictions`.
        test_predictions = None

        plt.plot(train_data, train_target, "bo", label="Train target")
        plt.plot(test_data, test_target, "ro", label="Test target")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend()
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return train_rmses, test_rmses

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
