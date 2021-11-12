#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
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
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--test", default=False, type=bool, help="Test flag")

# Settings
features_span = 5
features_mid = features_span

# Create features (vector of ord of letter and ords of nearby ones)
def create_features(data, span = 3, conversion = None):
    data_f = []
    for (i, dato) in enumerate(data):
        vect = [0] * (2*span+1)
        k = -1
        for j in range(i - span, i + span + 1):
            k = k + 1
            if j < 0 or j >= len(data): continue
            if conversion is not None: vect[k] = conversion(data[j])
            else: vect[k] = data[j]
        data_f.append(vect)
    return data_f

# Select just data with desired letter
def select_data(source, letter, letter_variants):
    result = types.SimpleNamespace()

    temp_data = []
    temp_target = []
    for (i, dato) in enumerate(source.data):
        if dato[features_mid] == ord(letter):
            temp_data.append(dato)
            temp_target.append(source.target[i])
    result.data = np.array(temp_data)
    result.target = np.array(temp_target)

    result.target = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore").fit_transform(np.reshape(result.target, (-1,1)))

    return result

def main(args: argparse.Namespace):

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        test = types.SimpleNamespace()

        # Split data for testing
        if args.test:
            size = len(train.data) // 3
            train.data, test.data, train.target, test.target =  train.data[size+1:], train.data[:size], train.target[size+1:], train.target[:size]

        # Normalize data
        train.data = train.data.lower()
        train.target = train.target.lower()

        # Create data features
        train.data = create_features(train.data, span=features_span, conversion=ord)
        train.target = [ord(t) for t in train.target]

        # Normalize and store original data and create features for testing
        if args.test:
            test.data = test.data.lower()
            test.target = test.target.lower()
        
            test_orig = types.SimpleNamespace()
            test_orig.data = test.data
            test_orig.target =  test.target

            test.data = create_features(test.data, span=features_span, conversion=ord)
            test.target = [ord(t) for t in test.target]

        # Dic of letters and its variants
        letters = { "a":"aá", "c":"cč", "d":"dď", "e":"eéě", "i":"ií", "n":"nň", "o":"oó", "r":"rř", "s":"sš", "t":"tť", "u":"uúů", "y":"yý", "z":"zž" }
        acc_total = 0

        # Letter to predict and its variants
        for letter in letters:
            letter_variants = letters[letter]

            # Select just data with desired letter
            train_s = select_data(train, letter, letter_variants)
            if args.test:
                test_s = select_data(test, letter, letter_variants)

            # Create model
            model = sklearn.pipeline.Pipeline([
                    ("StandardScaler", sklearn.preprocessing.StandardScaler()),
                    ("PolynomialFeature", sklearn.preprocessing.PolynomialFeatures(3, include_bias=True)),
                    ("MLP_classifier", sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation="relu", solver="adam", max_iter=1000, alpha=0.1, learning_rate="adaptive"))
                ])

            # Fit
            model.fit(train_s.data, train_s.target)

            # Predict probabs for testing
            if args.test:
                print("------", letter, "------")

                test_predictions = model.predict_proba(test_s.data)
                test_accuracy = sklearn.metrics.accuracy_score(np.argmax(test_s.target,axis=1), np.argmax(test_predictions,axis=1))
                print("TEST","Acc:",test_accuracy)
                acc_total = acc_total + test_accuracy

                # Get max prob
                test_predictions = np.argmax(test_predictions, axis=1)

                # Recerate original data
                test_orig.data = list(test_orig.data)
                k = 0
                for (i, l) in enumerate(test_orig.data):
                    if l == letter:
                        test_orig.data[i] = letter_variants[test_predictions[k]]
                        k = k + 1

                print("".join(test_orig.target[:200]))
                print("".join(test_orig.data[:200]))

        if args.test:
             print("TEST TOTAL Acc:", acc_total / len(letters))

        # Serialize the model if not testing
        if not args.test:
            with lzma.open(args.model_path, "wb") as model_file:
                pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = None

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
