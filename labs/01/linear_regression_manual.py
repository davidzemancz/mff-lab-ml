# fbc0b6cc-0238-11eb-9574-ea7484399335
# 7b885094-03f8-11eb-9574-ea7484399335

import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> float:
    # Load the Diabetes dataset
    dataset = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # Append a new feature to all input data, with value "1" (that will be multiplied by bias)
    dataset.data = np.append(dataset.data, np.ones([1,dataset.data.shape[0]]).transpose(), axis=1)
   
    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    # Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    optimal_weights = np.linalg.inv(train_data.transpose() @ train_data) @ train_data.transpose() @ train_target
    
    # Predict target values on the test set.
    # Compute root mean square error on the test set predictions.

    rmse = np.sqrt(np.mean(((test_data @ optimal_weights) - test_target)**2))

    return rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
