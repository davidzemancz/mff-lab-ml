# fbc0b6cc-0238-11eb-9574-ea7484399335
# 7b885094-03f8-11eb-9574-ea7484399335

import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=13, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> tuple[float, float]:
    # Load the Diabetes dataset
    dataset = sklearn.datasets.load_diabetes()

    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed) 

    lambdas = np.geomspace(0.01, 10, num=500)
       
    # L2 regularization, employing above defined lambdas.
    # For every model, compute the root mean squared error and return the
    # lambda producing lowest RMSE and the corresponding RMSE.
    best_lambda = -1
    best_rmse = 99999999 # Hoping my model gonna be better
    
    for l in lambdas:
        # lambda is hyperparameter -> not adapted by model itself

        model = sklearn.linear_model.Ridge(alpha = l)
        model = model.fit(train_data, train_target)
        prediction = model.predict(test_data)

        rmse = sklearn.metrics.mean_squared_error(test_target, prediction, squared=False)
        if rmse < best_rmse:
            best_rmse = rmse
            best_lambda = l

    # -------- Linear regression without L2 regularization -----------
    # model = sklearn.linear_model.LinearRegression()
    # model = model.fit(train_data, train_target)
    # rmse = sklearn.metrics.mean_squared_error(test_target, prediction, squared=False)
    # print(rmse)
    # ----------------------------------------------------------------

    if args.plot:
        # This block is not required to pass in ReCodEx, however, it is useful
        # to learn to visualize the results.

        # If you collect the respective results for `lambdas` to an array called `rmses`,
        # the following lines will plot the result if you add `--plot` argument.
        import matplotlib.pyplot as plt
        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength")
        plt.ylabel("RMSE")
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))
