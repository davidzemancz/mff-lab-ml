# fbc0b6cc-0238-11eb-9574-ea7484399335
# 7b885094-03f8-11eb-9574-ea7484399335

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
import sklearn.feature_selection
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.kernel_approximation
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

class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8-sig") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--test", default=False, type=bool, help="Test flag")

# Settings
features_span = 4
features_mid = features_span
letters = { "a":"aá", "c":"cč", "d":"dď", "e":"eéě", "i":"ií", "n":"nň", "o":"oó", "r":"rř", "s":"sš", "t":"tť", "u":"uúů", "y":"yý", "z":"zž" }
alphabet = list("aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž")

# Create one-hot features (one-hot vector of ord of letter and ords of nearby ones as indexies)
def create_features_oh(data, span = 3, conversion = None):
    data_f = []
    for (i, dato) in enumerate(data):
        vect = np.zeros([400]) #[0] * 400 #(2*span+1)
        vect[0] = conversion(data[i])
        for j in range(i - span, i + span + 1):
            dist = abs(j - i) if j - i != 0 else 1

            if j < 0 or j >= len(data): continue
            elif conversion is not None: vect[conversion(data[j])] = 1/dist
            else: vect[data[j]] = 1/dist

        data_f.append(vect)
    return np.array(data_f)

# Create features (vector of ord of letter and ords of nearby ones)
def create_features(data, span = 3, conversion = None):
    #return create_features_oh(data, span, conversion)
    
    data_f = []
    for (i, dato) in enumerate(data):
        vect = [0] * (2*span+1)
        k = -1
        vect[0] = conversion(data[i])
        for j in range(i - span, i + span + 1):
            k = k + 1

            if j < 0 or j >= len(data): continue
            elif conversion is not None: vect[k] = conversion(data[j])
            else: vect[k] = data[j]

        data_f.append(vect)
    return np.array(data_f)

# Select just data with desired letter form one-hot
def select_data_oh(source, letter, letter_variants):
    result = types.SimpleNamespace()

    temp_data = []
    temp_target = []
    for (i, dato) in enumerate(source.data):
        if dato[0] == ord(letter):
            dato[0] = 0
            temp_data.append(dato)
            temp_target.append(source.target[i])
    result.data = np.array(temp_data)
    result.target = np.array(temp_target)

    result.target = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore").fit_transform(np.reshape(result.target, (-1,1)))

    return result

# Select just data with desired letter
def select_data(source, letter, letter_variants):
    #return select_data_oh(source, letter, letter_variants)
    
    result = types.SimpleNamespace()

    temp_data = []
    temp_target = []
    for (i, dato) in enumerate(source.data):
        if dato[features_mid] == ord(letter):
            temp_data.append(dato)
            temp_target.append(source.target[i])
    result.data = np.array(temp_data)
    result.target = np.array(temp_target)
    
    #result.target = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore").fit_transform(np.reshape(result.target, (-1,1)))

    return result

# Accuray
def accuracy(gold: str, system: str) -> float:
    gold, system = gold.split(), system.split()
    assert len(gold) == len(system), "The gold and system outputs must have same number of words: {} vs {}.".format(len(gold), len(system))

    words, correct = 0, 0
    for gold_token, system_token in zip(gold, system):
        words += 1
        match = gold_token == system_token
        if not match:
            print(gold_token, "x", system_token)
        correct += match

    return correct / words

# Levenstein distance
def levenshtein(word1, word2):
    size_x = len(word1) + 1
    size_y = len(word2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if word1[x-1] == word2[y-1]:
                matrix [x,y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1], matrix[x, y-1] + 1)
            else:
                matrix [x,y] = min(matrix[x-1,y] + 1, matrix[x-1,y-1] + 1,matrix[x,y-1] + 1)
    return (matrix[size_x - 1, size_y - 1])

def main(args: argparse.Namespace):

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        dictionary = Dictionary()
        test = types.SimpleNamespace()

        # Split data for testing
        if args.test:
            size = len(train.data) // 2
            train.data, test.data, train.target, test.target =  train.data[size+1:], train.data[:size], train.target[size+1:], train.target[:size]

        # Normalize data
        train.data = train.data.lower()
        train.target = train.target.lower()

        for word in dictionary.variants:
            for variant in dictionary.variants[word]:
                train.data =  train.data + "     " + word
                train.target =  train.target + "     " + variant


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
            if args.test:
                test_s = select_data(test, letter, letter_variants)

            # Create model
            if not args.test: print("------", letter, "------")
            model = sklearn.pipeline.Pipeline(steps = [
                    ("OneHotEncoder", sklearn.preprocessing.OneHotEncoder(categories="auto", sparse=True, handle_unknown="ignore")),
                    ("PolynomialFeatures", sklearn.preprocessing.PolynomialFeatures(3, include_bias=True, interaction_only=True)),
                    ("VarianceThreshold", sklearn.feature_selection.VarianceThreshold()),
                    #("MLPClassifier", sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100), activation="relu", solver="adam", max_iter=50, alpha=0.1, learning_rate="adaptive", tol=0.001, verbose=True))
                    #("LogisticRegression", sklearn.linear_model.LogisticRegression(solver="saga", multi_class="multinomial", max_iter=200, verbose=True, n_jobs=-1))
                    ("SGDClassifier", sklearn.linear_model.SGDClassifier(verbose=True, n_jobs=-1, loss="log"))
                ])

            # Fit
            model.fit(train_s.data, train_s.target)

            # Reduce model size
            est = model.get_params()["steps"][-1][1]
            #est.sparsify()
            for i in range(len(est.coef_)): est.coef_[i] = est.coef_[i].astype(np.float16)
            for i in range(len(est.intercept_)): est.intercept_[i] = est.intercept_[i].astype(np.float16)

            # Store model in dic
            model_dic[letter] = model
            
            # Predict probabs for testing
            if args.test:
                print("------", letter, "------")

                test_predictions = model.predict_proba(test_s.data)
                #test_accuracy = sklearn.metrics.accuracy_score(np.argmax(test_s.target,axis=1), np.argmax(test_predictions,axis=1))
                #print("TEST","Acc:",test_accuracy)
                #acc_total = acc_total + test_accuracy

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
            orig_list = "".join(test_orig.data).split()
            result_list = "".join(test_result).split()

            for (i, word_no_dia), (j, word) in zip(enumerate(orig_list), enumerate(result_list)):
                if word_no_dia in dictionary.variants:
                    match = False
                    min_lev_dist = 9999999
                    min_lev_dist_index = -1
                    for (k, variant) in enumerate(dictionary.variants[word_no_dia]):
                        
                        lev_dist = levenshtein(word, variant)
                        if lev_dist < min_lev_dist: 
                            min_lev_dist = lev_dist
                            min_lev_dist_index = k

                        if variant == word:
                            match = True
                            break
                    
                    if not match and min_lev_dist_index > -1:
                        result_list[j] = dictionary.variants[word_no_dia][min_lev_dist_index]

            print(accuracy("".join(test_orig.target), " ".join(result_list)))

        # Serialize the model_dic if not testing
        if not args.test:
            with lzma.open(args.model_path, "wb") as model_file:
                pickle.dump(model_dic, model_file)

    else:
        if args.test:
            train = Dataset()
            dictionary = Dictionary()
            test = types.SimpleNamespace()
            test_orig = types.SimpleNamespace()

            size = len(train.data) // 2
            train.data, test.data, train.target, test.target =  train.data[size+1:], train.data[:size], train.target[size+1:], train.target[:size]

            test_orig.data = test.data
            test_orig.target =  test.target
        else:
            test = Dataset(args.predict)
            dictionary = Dictionary()

        # Prepare data for result
        test_result = list(test.data)

        # Store original data
        test_orig = types.SimpleNamespace()
        test_orig.data = test.data
        test_orig.target = test.target

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


        orig_list = "".join(test_orig.data).split()
        result_list = "".join(test_result).split()

        for (i, word_no_dia), (j, word) in zip(enumerate(orig_list), enumerate(result_list)):
            word_no_dia = word_no_dia.lower()
            word = word.lower()
            if word_no_dia in dictionary.variants:
                match = False
                min_lev_dist = 9999999
                min_lev_dist_index = -1
                for (k, variant) in enumerate(dictionary.variants[word_no_dia]):
                    lev_dist = levenshtein(word, variant)
                    if lev_dist < min_lev_dist: 
                        min_lev_dist = lev_dist
                        min_lev_dist_index = k

                    if variant == word:
                        match = True
                        break
                
                if not match and min_lev_dist_index > -1:
                    #print(result_list[j], "-", dictionary.variants[word_no_dia][min_lev_dist_index])
                    capitalize = result_list[j][0].isupper()
                    #print(result_list[j], "-", dictionary.variants[word_no_dia])
                    result_list[j] = dictionary.variants[word_no_dia][min_lev_dist_index]
                   
                    if capitalize: 
                        result_list[j] = result_list[j].capitalize()

        # Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = " ".join(result_list)

        if args.test:
            print(accuracy("".join(test_orig.target), predictions))

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
