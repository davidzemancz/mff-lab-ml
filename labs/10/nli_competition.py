# fbc0b6cc-0238-11eb-9574-ea7484399335
# 7b885094-03f8-11eb-9574-ea7484399335

import argparse
import lzma
import pickle
import os

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
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name="nli_dataset.train.txt"):
        if not os.path.exists(name):
            raise RuntimeError("The {} was not found, please download it from ReCodEx".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")
parser.add_argument("--test", default=False, type=bool, help="Test flag")

def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        model = sklearn.pipeline.Pipeline([
            ("FeatureUnion", sklearn.pipeline.FeatureUnion([
                 ("TfidfVectorizer_char", sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, analyzer="char_wb", ngram_range=(1,4), max_features=10000)),
                 ("TfidfVectorizer_word", sklearn.feature_extraction.text.TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1,3), max_features=10000)),
                 #("HashingVectorizer", sklearn.feature_extraction.text.HashingVectorizer(lowercase=False, ngram_range=(1,4)))
             ], n_jobs=-1, verbose=True)),
            ("SGDClassifier", sklearn.linear_model.SGDClassifier(verbose=True, n_jobs=-1, loss="hinge"))
        ])

        if args.test:
            scores = sklearn.model_selection.cross_val_score(model, train.data, train.target, cv=3, n_jobs=-1)
            print("Cross-validation with 3 folds: {:.2f} +-{:.2f}".format(100 * scores.mean(), 100 * scores.std()))
        else:
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
