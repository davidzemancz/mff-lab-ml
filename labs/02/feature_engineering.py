#!/usr/bin/env python
import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--debug", default=True, type=bool, help="Print intermediate results for debugging")
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def get_int_cols(data): 
    """ Find columns that have only int values. Returns tuple, first is list of int-only columns indexies, second is suplement to all column indexies. """
    
    # Find and store int cols to dictionary
    int_cols_dic = {}
    for row in range(data.shape[0]): 
        for col in range(data.shape[1]):
            val = data[row,col]
            if val is int or val.is_integer():
                if int_cols_dic.get(col) is None:
                    int_cols_dic[col] = True
            else:
                int_cols_dic[col] = False

    # Push cols with int values to list
    int_cols = []
    non_int_cols = []
    for key in int_cols_dic:
        if int_cols_dic.get(key): int_cols.append(key)
        else: non_int_cols.append(key)

    return int_cols, non_int_cols

def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    # Data and targets
    data,target = dataset.data, dataset.target
    
    # Process the input columns in the following way:
    #
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of an exercise). Encode the values with one-hot encoding
    #   using `sklearn.preprocessing.OneHotEncoder` (note that its output is by
    #   default sparse, you can use `sparse=False` to generate dense output;
    #   also use `handle_unknown="ignore"` to ignore missing values in test set).
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; use `sklearn.preprocessing.StandardScaler`.
    #
    # In the output, there should be first all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.
    # 
    # To the current features, append polynomial features of order 2.
    # If the input values are [a, b, c, d], you should append
    # [a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]. You can generate such polynomial
    # features either manually, or using
    # `sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)`.

    # You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.

    # Get columns that have only int values to list
    int_cols, non_int_cols = get_int_cols(data)
   
    # OneHotEncoder to fit int features to one-hot encoding ... replace by ColumnTransformer
    if False and len(int_cols) > 0:
        # Using OneHotEncoder
        enc = sklearn.preprocessing.OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore")
        # First need to fit
        enc.fit(data[:, int_cols])
        # Then transform original data to trensformed data with one-hot features
        data_int_t = enc.transform(data[:, int_cols])

    # Normalize non int columns ... replace by ColumnTransformer
    if False and len(non_int_cols) > 0:
        # Using StandardScaler
        enc = sklearn.preprocessing.StandardScaler()
        # First need to fit
        enc.fit(data[:, non_int_cols])
        # Then transform original data to trensformed data with one-hot features
        data_non_int_t = enc.transform(data[:, non_int_cols])

    # Or rather use general column transformer instead separated transformes
    ctr = sklearn.compose.ColumnTransformer([
        ("OneHotEncoder for int cols", sklearn.preprocessing.OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore"), int_cols),
        ("StandardScaler for non int cols", sklearn.preprocessing.StandardScaler(), non_int_cols),
        ("PolynomialFeatures for all cols", sklearn.preprocessing.PolynomialFeatures(2, include_bias=False), int_cols + non_int_cols),
        ],n_jobs=-1)
    # Transformed data
    data_t = ctr.fit_transform(data)

    # --- DEBUG ---
    #if args.debug: print(data[0])
    #if args.debug: print(data_t[0])

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data_t, target, test_size=args.test_size, random_state=args.seed)

    # Fit the feature processing steps on the training data.
    # Then transform the training data into `train_data` (you can do both these
    # steps using `fit_transform`), and transform testing data to `test_data`.
    # ... hmm, ok
    # ...

    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            if not args.debug: print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))))
