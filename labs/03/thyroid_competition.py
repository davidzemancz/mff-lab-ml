# fbc0b6cc-0238-11eb-9574-ea7484399335
# 7b885094-03f8-11eb-9574-ea7484399335

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

class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")
parser.add_argument("--test", default=False, type=bool, help="Test flag")

def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        test = types.SimpleNamespace()

        if args.test:
            train.data, test.data, train.target, test.target = sklearn.model_selection.train_test_split(train.data, train.target, test_size=0.5, random_state=42)

        #  Train a model on the given dataset and store it in `model`.
        model = sklearn.pipeline.Pipeline([
            ("Transform - Column tranformer",  sklearn.compose.ColumnTransformer([
                ("OneHotEncoder", sklearn.preprocessing.OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore"), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]),
                #("StandardScaler for non int cols", sklearn.preprocessing.StandardScaler(), [15,16,17,18,19,20]),
                ("RobustScaler for non int cols", sklearn.preprocessing.RobustScaler(), [15,16,17,18,19,20]),
                ], n_jobs=-1)), 
            ("Transform - PolynomialFeatures for all cols", sklearn.preprocessing.PolynomialFeatures(3, include_bias=True)),
            ("Estimator - Logistic regression", sklearn.linear_model.LogisticRegression(max_iter = 1000))]
        )


        # Transform data with transformers and than use estimator
        model.fit(train.data, train.target)
        
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

        # Test on test data
        if args.test:
            train_predictions = model.predict(train.data)
            train_loss = sklearn.metrics.log_loss(train.target, train_predictions)
            train_accuracy = sklearn.metrics.accuracy_score(train.target, sklearn.preprocessing.binarize(train_predictions.reshape(-1, 1), threshold=0.5))
            print("TRAIN","Loss:",train_loss,"Acc:",train_accuracy)

            test_predictions = model.predict(test.data)
            test_loss = sklearn.metrics.log_loss(test.target, test_predictions)
            test_accuracy = sklearn.metrics.accuracy_score(test.target, sklearn.preprocessing.binarize(test_predictions.reshape(-1, 1), threshold=0.5))
            print("TEST","Loss:",test_loss,"Acc:",test_accuracy)
        
        

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
