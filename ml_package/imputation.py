from ml_package import *


class Impute_Transformer:
    def __init__(self):
        pass

    def fit(self, X_train, numImputer, catImputer, num_columns=None, obj_columns=None):
        try:
            if not num_columns:
                num_columns = X_train.select_dtypes("float64").columns
        except ValueError:
            pass
        try:
            if not obj_columns:
                obj_columns = X_train.select_dtypes(include=["object", "int64"]).columns
        except ValueError:
            pass

        # Set the class variables
        self.num_columns = num_columns
        self.obj_columns = obj_columns
        self.numImputer = numImputer
        self.catImputer = catImputer
        # Fit the variables
        self.numImputer.fit(X_train[num_columns])
        self.catImputer.fit(X_train[obj_columns])

    def transform(self, X):
        # Impute the columns
        X.loc[:, self.num_columns] = self.numImputer.transform(X[self.num_columns])
        X.loc[:, self.obj_columns] = self.catImputer.transform(X[self.obj_columns])

        return X

    def fit_transform(
        self, X, numImputer, catImputer, num_columns=None, obj_columns=None
    ):
        self.fit(X, numImputer, catImputer, num_columns=None, obj_columns=None)
        return self.transform(X)


def impute_missing(
    X_train, X_test, numImputer, catImputer, num_columns=None, obj_columns=None
):
    X_train = X_train.copy()
    X_test = X_test.copy()

    try:
        if not num_columns:
            num_columns = X_train.select_dtypes("float64").columns
    except ValueError:
        pass
    try:
        if not obj_columns:
            obj_columns = X_train.select_dtypes(include=["object", "int64"]).columns
    except ValueError:
        pass

    # Do mean imputations for numeric columns
    X_train.loc[:, num_columns] = numImputer.fit_transform(X_train[num_columns])
    X_test.loc[:, num_columns] = numImputer.transform(X_test[num_columns])
    # Do freq imputations for categoric columns
    X_train.loc[:, obj_columns] = catImputer.fit_transform(X_train[obj_columns])
    X_test.loc[:, obj_columns] = catImputer.fit_transform(X_test[obj_columns])

    return X_train, X_test
