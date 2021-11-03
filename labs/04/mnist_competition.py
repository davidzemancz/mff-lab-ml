#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

import types

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.neural_network

class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")
parser.add_argument("--test", default=False, type=bool, help="Test flag")

def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        test = types.SimpleNamespace()

        if args.test:
            train.data = train.data[1000:]
            train.target = train.target[1000:]
            train.data, test.data, train.target, test.target = sklearn.model_selection.train_test_split(train.data, train.target, test_size=0.5, random_state=42)
        
        # Train a model on the given dataset and store it in `model`.
        model = sklearn.pipeline.Pipeline([
            ("Estimator - MLP classifier", sklearn.neural_network.MLPClassifier(activation="relu", solver="sgd", max_iter=200))]
        )

        # Transform data with transformers and than use estimator
        model.fit(train.data, train.target)

        # Test on test data
        if args.test:
            train_predictions = model.predict(train.data)
            train_loss = sklearn.metrics.log_loss(train.target, train_predictions)
            train_accuracy = sklearn.metrics.accuracy_score(train.target, train_predictions)
            print("TRAIN","Loss:",train_loss,"Acc:",train_accuracy)

            test_predictions = model.predict(test.data)
            test_loss = sklearn.metrics.log_loss(test.target, test_predictions)
            test_accuracy = sklearn.metrics.accuracy_score(test.target, test_predictions)
            print("TEST","Loss:",test_loss,"Acc:",test_accuracy)

        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLPClassifier is in `mlp` variable.
        # mlp._optimizer = None
        # for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        # for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
