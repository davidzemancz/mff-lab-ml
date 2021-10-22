import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing

import numpy as np

class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rentals in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
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
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")

def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        
        #test = Dataset()
        #train.data, test.data, train.target, test.target = sklearn.model_selection.train_test_split(train.data, train.target, test_size=0.5) 

        # Train a model on the given dataset and store it in `model`.
        pipeline = sklearn.pipeline.Pipeline([
            ("Column tranformer",  sklearn.compose.ColumnTransformer([
                ("OneHotEncoder", sklearn.preprocessing.OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore"), [0,1,2,3,4,5,6,7]),
                ("StandardScaler for non int cols", sklearn.preprocessing.StandardScaler(), [8,9,10,11]),
                ], n_jobs=-1)), 
            ("PolynomialFeatures for all cols", sklearn.preprocessing.PolynomialFeatures(2, include_bias=True))]
        )

        train_data = pipeline.fit_transform(train.data)
        model = sklearn.linear_model.SGDRegressor()
        model = model.fit(train_data, train.target)
        
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as file:
            pickle.dump(model, file)

        # Serialize preprocessing pipeline
        with lzma.open("rental_competition.prc", "wb") as file:
            pickle.dump(pipeline, file)

        #rmse = sklearn.metrics.mean_squared_error(test.target, predictions, squared=False)
        #print(rmse)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as file:
            model = pickle.load(file)

        with lzma.open("rental_competition.prc", "rb") as file:
            pipeline = pickle.load(file)

        # Generate `predictions` with the test set predictions.
        test_data = pipeline.transform(test.data)
        predictions = model.predict(test_data)

        return predictions

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
