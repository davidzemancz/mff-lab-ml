import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
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

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    # Get columns that have only int values to list
    int_cols, non_int_cols = get_int_cols(train_data)
   
    # Use general column transformer instead separated transformes to specific columns
    ctr = sklearn.compose.ColumnTransformer([
        ("OneHotEncoder for int cols", sklearn.preprocessing.OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore"), int_cols),
        ("StandardScaler for non int cols", sklearn.preprocessing.StandardScaler(), non_int_cols),
        ], n_jobs=-1)

    # Put column transforem and PolynomialFeatures to pipline in appropriate order
    pipeline = sklearn.pipeline.Pipeline([
        ("Column tranformer", ctr), 
        ("PolynomialFeatures for all cols", sklearn.preprocessing.PolynomialFeatures(2, include_bias=False))])    

    # Fit and transform train data
    train_data = pipeline.fit_transform(train_data)

    # Transform test data ... JUST transform, NO fitting
    test_data = pipeline.transform(test_data)

    return train_data[:5], test_data[:5]
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(";".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))))

