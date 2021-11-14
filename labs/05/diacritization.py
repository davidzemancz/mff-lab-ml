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
from joblib import parallel_backend

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
features_span = 3
features_mid = features_span
letters = { "a":"aá", "c":"cč", "d":"dď", "e":"eéě", "i":"ií", "n":"nň", "o":"oó", "r":"rř", "s":"sš", "t":"tť", "u":"uúů", "y":"yý", "z":"zž" }
alphabet = list("aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž")

# Create features (vector of ord of letter and ords of nearby ones)
def create_features(data, span = 3, conversion = None):
    data_f = []
    for (i, dato) in enumerate(data):
        vect = [0] * (2*span+1)
        k = -1
        for j in range(i - span, i + span + 1):
            k = k + 1
            if j < 0 or j >= len(data): continue
            elif conversion is not None: vect[k] = conversion(data[j])
            else: vect[k] = data[j]

            #dist = abs(j - i) if j - i != 0 else 1
            #vect[k] = vect[k] * (1/dist)

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
            size = len(train.data) // 2
            train.data, test.data, train.target, test.target =  train.data[size+1:], train.data[:size], train.target[size+1:], train.target[:size]

        # Normalize data
        train.data = train.data.lower()
        train.target = train.target.lower()

        # Create data features
        train.data = create_features(train.data, span=features_span, conversion=ord)
        #train.data = create_features_oh(train.data)
        train.target = [ord(t) for t in train.target]

        # Normalize and store original data and create features for testing
        if args.test:
            test.data = test.data.lower()
            test.target = test.target.lower()
        
            test_orig = types.SimpleNamespace()
            test_orig.data = test.data
            test_orig.target =  test.target

            test_result = list(test.data)

            test.data = create_features(test.data, span=features_span, conversion=ord)
            #test.data = create_features_oh(test.data)
            test.target = [ord(t) for t in test.target]

        # Dic of letters and its variants
        acc_total = 0

        model_dic = {}
        # Letter to predict and its variants
        for letter in letters:
            letter_variants = letters[letter]

            # Select just data with desired letter
            train_s = select_data(train, letter, letter_variants)
            #train_s = select_data_oh(train, letter)
            if args.test:
                test_s = select_data(test, letter, letter_variants)
                #test_s = select_data_oh(test, letter)

            # Create model
            print("------", letter, "------")
            model = sklearn.pipeline.Pipeline(steps = [
                    ("PolynomialFeatures", sklearn.preprocessing.PolynomialFeatures(2, include_bias=True, interaction_only=True)),
                    #("StandardScaler", sklearn.preprocessing.StandardScaler()),
                    ("OneHotEncoder", sklearn.preprocessing.OneHotEncoder(categories="auto", sparse=True, handle_unknown="ignore", drop="first")),
                    ("MLPClassifier", sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50), activation="relu", solver="adam", max_iter=200, alpha=0.1, learning_rate="adaptive", tol=0.001, verbose=True))
                ])

            # Fit
            model.fit(train_s.data, train_s.target)

            # Reduce model size
            mlp = model.get_params()["steps"][-1][1]
            mlp._optimizer = None
            for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
            for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

            # Store model in dic
            model_dic[letter] = model
            
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
                k = 0
                for (i, l) in enumerate(test_result):
                    if l == letter:
                        test_result[i] = letter_variants[test_predictions[k]]
                        k = k + 1

                #print("".join(test_orig.target[:200]))
                #print("".join(test_result[:200]))


        if args.test:
            total = 0
            correct = 0
            for (i, letter) in enumerate(test_orig.data):
                if letter in letters:
                    total = total + 1
                    if test_result[i] == test_orig.target[i]:
                        correct = correct + 1
        
        if args.test:
             print("TEST TOTAL Acc:", correct / total)

        # Serialize the model_dic if not testing
        if not args.test:
            with lzma.open(args.model_path, "wb") as model_file:
                pickle.dump(model_dic, model_file)

    else:
        if args.test:
            train = Dataset()
            test = types.SimpleNamespace()
            test_orig = types.SimpleNamespace()

            size = len(train.data) // 2
            train.data, test.data, train.target, test.target =  train.data[size+1:], train.data[:size], train.target[size+1:], train.target[:size]

            test_orig.data = test.data
            test_orig.target =  test.target
        else:
            test = Dataset(args.predict)

        # Prepare data for result
        test_result = list(test.data)

        # Normalize data
        test.data = test.data.lower()
        test.target = test.target.lower()

        # Create features
        test.data = create_features(test.data, span=features_span, conversion=ord)
        test.target = [ord(t) for t in test.target]

        # Load dic of models (letter is key)
        with lzma.open(args.model_path, "rb") as model_file:
            model_dic = pickle.load(model_file)

        for letter in letters:
            letter_variants = letters[letter]

            # Select just data with desired letter
            test_s = select_data(test, letter, letter_variants)

            # Get model for letter
            model = model_dic[letter]

            # Predict and get max prob
            test_predictions = model.predict_proba(test_s.data)
            test_predictions = np.argmax(test_predictions, axis=1)

            # Recerate original data
            k = 0
            for (i, l) in enumerate(test_result):
                if l == letter:
                    test_result[i] = letter_variants[test_predictions[k]]
                    k = k + 1
                elif l == letter.upper():
                    test_result[i] = letter_variants[test_predictions[k]].upper()
                    k = k + 1


        # Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = "".join(test_result)

        if args.test:
            total = 0
            correct = 0
            for (i, letter) in enumerate(test_orig.data):
                if letter in letters:
                    total = total + 1
                    if test_result[i] == test_orig.target[i]:
                        correct = correct + 1

            #print("".join(test_orig.target[:200]))
            #print("".join(test_result[:200]))

            print("TEST TOTAL Acc:", correct / total)

       


        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
