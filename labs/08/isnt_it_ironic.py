import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np

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
    def __init__(self,
                 name="isnt_it_ironic.train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            for line in dataset_file:
                label, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.target.append(int(label))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")
parser.add_argument("--test", default=False, type=bool, help="Test flag")

def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        
        model = sklearn.pipeline.Pipeline([
            ("TfidfVectorizer", sklearn.feature_extraction.text.TfidfVectorizer(strip_accents="unicode", max_features=20000, analyzer="char_wb", stop_words="english", ngram_range=(1,3))),
            #("CountVectorizer", sklearn.feature_extraction.text.CountVectorizer(strip_accents="unicode", max_features=10000, analyzer="char_wb", stop_words="english", ngram_range=(1,3))),
            #("HashingVectorizer", sklearn.feature_extraction.text.HashingVectorizer(strip_accents="unicode", analyzer="char_wb", ngram_range=(1,3), binary=True)),
            ("MultinomialNB", sklearn.naive_bayes.MultinomialNB(alpha=0.5)),
            #("LogisticRegression", sklearn.linear_model.LogisticRegression())
        ])

        if args.test:
            scores = sklearn.model_selection.cross_val_score(model, train.data, train.target, scoring="f1", cv=5)
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

        # Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
