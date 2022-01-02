# fbc0b6cc-0238-11eb-9574-ea7484399335
# 7b885094-03f8-11eb-9574-ea7484399335


import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
import pandas as pd

import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
import sklearn.feature_selection
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.kernel_approximation
import sklearn.ensemble
import sklearn.feature_extraction
import sklearn.naive_bayes


class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(self,
                 name="human_activity_recognition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and if it contains column "class", split it to `targets`.
        self.data = pd.read_csv(name)
        if "class" in self.data:
            self.target = np.array([Dataset.CLASSES.index(target) for target in self.data["class"]], np.int32)
            self.data = self.data.drop("class", axis=1)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="human_activity_recognition.model", type=str, help="Model path")
parser.add_argument("--test", default=False, type=bool, help="Test flag")

def main(args: argparse.Namespace):
    args.model_path = "GradientBoostingClassifier." + args.model_path
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        if args.test:
            for ests in [300,500,1000,3000,10000]:
                for depth in [16,24,32]:
                    for crit in ["entropy","gini"]:
                        #for ss in [0.3,0.5,1]:
                        model = sklearn.pipeline.Pipeline([
                            ("StandardScaler", sklearn.preprocessing.StandardScaler()),
                            #("GradientBoostingClassifier", sklearn.ensemble.GradientBoostingClassifier(n_estimators=ests, max_depth=depth, subsample=ss, verbose=False)),
                            ("RandomForestClassifier", sklearn.ensemble.RandomForestClassifier(n_estimators=ests, criterion=crit, max_depth=depth, verbose=False, n_jobs=-1)),
                        ])
                        scores = sklearn.model_selection.cross_val_score(model, train.data, train.target, cv=3, n_jobs=-1)
                        print("Estimators:", ests, "| Crit:", crit, "| Depth:", depth)
                        print("Cross-validation with 5 folds: {:.2f} +-{:.2f}".format(100 * scores.mean(), 100 * scores.std()))
        else:
            model = sklearn.pipeline.Pipeline([
                    ("StandardScaler", sklearn.preprocessing.StandardScaler()),
                    ("GradientBoostingClassifier", sklearn.ensemble.GradientBoostingClassifier(n_estimators=3000, max_depth=8, subsample=0.5, verbose=False)),
                ])

            scores = sklearn.model_selection.cross_val_score(model, train.data, train.target, cv=3, n_jobs=-1)
            print("Cross-validation with 5 folds: {:.2f} +-{:.2f}".format(100 * scores.mean(), 100 * scores.std()))

            # Train a model on the given dataset and store it in `model`.
            model.fit(train.data, train.target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
