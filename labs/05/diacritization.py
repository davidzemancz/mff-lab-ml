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

def create_features(data, span = 3):
    for (i, dato) in enumerate(data):
        vect = np.zeros([2*span+2])
        k = 0
        for j in range(i - span, i + span + 1):
            if j < 0 or j >= len(data): continue 
            vect[k] = data[j]
            k = k + 1
        data_f = vect
    return data_f
    

def main(args: argparse.Namespace):

    print(create_features([[0],[1],[2],[3]], span=1))

    return

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        test = types.SimpleNamespace()
       
        if args.test:
            size = len(train.data) // 2
            train.data, test.data, train.target, test.target = train.data[:size], train.data[size+1:], train.target[:size], train.target[size+1:]

        temp_data = np.zeros((len(train.data), 5))
        temp_target = np.zeros((len(train.data),2))
        k = 0
        for (i, dato) in enumerate(train.data):
            if train.data[i].lower() == "a":
                k = k + 1
                temp_data[k] = np.array([ord(train.data[i - 2]) if i - 2 >= 0 else 0, ord(train.data[i - 1])  if i - 1 >= 0 else 0, ord(train.data[i]), ord(train.data[i + 1]) if i + 1 < len(train.data) else 0, ord(train.data[i + 2])  if i + 2 < len(train.data) else 0])
                oh = [1,0] if train.target[i] == "a" else [0,1]
                temp_target[k] = np.array(oh)
        train.data = temp_data[:k]
        train.target = temp_target[:k]


        temp_data = np.zeros((len(test.data), 5))
        temp_target = np.zeros((len(test.data),2))
        k = 0
        for (i, dato) in enumerate(test.data):
            if test.data[i].lower() == "a":
                k = k + 1
                temp_data[k] = np.array([ord(test.data[i - 2]) if i - 2 >= 0 else 0, ord(test.data[i - 1])  if i - 1 >= 0 else 0, ord(test.data[i]), ord(test.data[i + 1]) if i + 1 < len(test.data) else 0, ord(test.data[i + 2])  if i + 2 < len(test.data) else 0])
                oh = [1,0] if test.target[i] == "a" else [0,1]
                temp_target[k] = np.array(oh)
        test.data = temp_data[:k]
        test.target = temp_target[:k]

        model = sklearn.pipeline.Pipeline([
                ("StandardScaler", sklearn.preprocessing.StandardScaler()),
                ("MLP_classifier", sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(500), activation="relu", solver="adam", max_iter=1000, alpha=0.1, learning_rate="adaptive"))
            ])

        # Fit
        model.fit(train.data, train.target)

        test_predictions = model.predict_proba(test.data)
        test_accuracy = sklearn.metrics.accuracy_score(np.argmax(test.target,axis=1), np.argmax(test_predictions,axis=1))
        print("TEST","Acc:",test_accuracy)

        # Serialize the model.
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
