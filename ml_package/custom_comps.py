from ml_package import *
from sklearn.decomposition import PCA
import math


class PCA_Transformer:
    def __init__(self, n_components=0.8, svd_solver="full"):
        self.n_components = n_components
        self.svd_solver = svd_solver

    def fit(self, X_train, groups):
        self.groups = groups
        self.pcas = {}
        for k, v in self.groups.items():
            self.pcas[k] = PCA(
                n_components=self.n_components, svd_solver=self.svd_solver
            )
            self.pcas[k].fit(X_train[v])

    def transform(self, X):
        X = X.copy()
        for k, v in self.groups.items():
            tmp = X[v]
            X.drop(columns=v, inplace=True)
            for i in range(self.pcas[k].n_components_):
                X[f"{k}_PC_{i}"] = self.pcas[k].transform(tmp)[:, i]
        return X

    def fit_transform(self, X_train, groups):
        self.fit(X_train, groups)
        return self.transform(X_train)


class Center_Transformer:
    def __init__(self):
        pass

    def fit(self, X_train, num_columns):
        self.num_columns = num_columns
        self.means = {}
        for c in self.num_columns:
            self.means[c] = X_train[c].mean()

    def transform(self, X):
        X = X.copy()
        for c in self.num_columns:
            X[c] = X[c] - self.means[c]
        return X

    def fit_transform(self, X_train, num_columns):
        self.fit(X_train, num_columns)
        return self.transform(X_train)


class Scaling_Transformer:
    def __init__(self):
        pass

    def fit(self, X_train, scaler, num_columns):
        self.scaler = scaler
        self.num_columns = num_columns

        self.scaler.fit(X_train[self.num_columns])

    def transform(self, X):
        X[self.num_columns] = self.scaler.transform(X[self.num_columns])
        return X

    def fit_transform(self, X_train, scaler, num_columns):
        self.fit(X_train, scaler, num_columns)
        return self.transform(X_train)


class Center_Scale_Transform:
    def __init__(self):
        pass

    def fit(self, X, scaler, pow_transformer, num_columns):
        self.ct = Center_Transformer()
        self.scaler = scaler
        self.pow_transformer = pow_transformer
        self.num_columns = num_columns
        X = X.copy()
        X = self.ct.fit_transform(X, num_columns)
        self.scaler.fit(X[num_columns])
        self.pow_transformer.fit(X[num_columns])

    def transform(self, X):
        X = X.copy()
        X[self.num_columns] = self.ct.transform(X[self.num_columns])
        X[self.num_columns] = self.scaler.transform(X[self.num_columns])
        X[self.num_columns] = self.pow_transformer.transform(X[self.num_columns])
        return X

    def fit_transform(self, X, scaler, pow_transformer, num_columns):
        self.fit(X, scaler, pow_transformer, num_columns)
        return self.transform(X)


def PCATransformer(x_train, x_test, groups):
    pca = PCA(n_components=0.8, svd_solver="full")
    x_train = x_train.copy()
    x_test = x_test.copy()
    for k, v in groups.items():
        tmp_train = x_train[v]
        tmp_test = x_test[v]
        x_train.drop(columns=v, inplace=True)
        x_test.drop(columns=v, inplace=True)
        pca.fit(tmp_train)
        for i in range(pca.n_components_):
            x_train[f"{k}_PC_{i}"] = pca.transform(tmp_train)[:, i]
            x_test[f"{k}_PC_{i}"] = pca.transform(tmp_test)[:, i]

    return (x_train, x_test)


def CenterTransformer(x_train, x_test, num_columns):
    x_train = x_train.copy()
    x_test = x_test.copy()
    for c in num_columns:
        mean = x_train[c].mean()
        x_train[c] = x_train[c] - mean
        x_test[c] = x_test[c] - mean

    return (x_train, x_test)


def CenterScaleTransform(x_train, x_test, scaler, pow_transformer, num_columns=None):
    if not num_columns:
        num_columns = x_train.select_dtypes(exclude=["object"]).columns
    x_train, x_test = CenterTransformer(x_train, x_test, num_columns)

    x_train[num_columns] = scaler.fit_transform(x_train[num_columns])
    x_test[num_columns] = scaler.transform(x_test[num_columns])

    x_train[num_columns] = pow_transformer.fit_transform(x_train[num_columns])
    x_test[num_columns] = pow_transformer.transform(x_test[num_columns])

    return x_train, x_test


def CorrelationClustering(X, columns=None, threshold=0.2):
    if not columns:
        col_details = data_prep.get_column_types(X)
        columns = col_details["Continious"] + col_details["Discrete"]
    corr_mat = X[columns].corr()
    distance_mat = 1 - np.abs(corr_mat)
    return AgglomerativeClustering(distance_mat, threshold)


def AgglomerativeClustering(distance_mat, threshold=0.2):
    groups = {}
    np.fill_diagonal(distance_mat.values, 1)
    values = np.sort((np.unique(distance_mat.values)))
    values = values[values < threshold]

    for v in values:
        mat_filter = np.where(distance_mat == v)
        col_filter = np.unique(mat_filter[0], return_counts=True)

        if (len(col_filter[0]) - 1) != max(col_filter[1]):
            print("Multiple Clusters")
            raise Exception

        row_filter = np.unique(mat_filter[1], return_counts=True)
        cols = list(distance_mat.iloc[row_filter[0], col_filter[0]].columns)
        k = " ".join(cols)
        added_list = []
        keys = list(groups.keys())
        for key in keys:
            for c in cols:
                if c in key:
                    if len(added_list):
                        groups[added_list[0]].update(groups[added_list[0]])
                        new_key = " ".join(groups[key])
                        groups[new_key] = groups.pop(key)
                        groups.pop(added_list[0])
                        added_list = [new_key]
                    else:
                        groups[key].update(cols)
                        new_key = " ".join(groups[key])
                        groups[new_key] = groups.pop(key)
                        added_list.append(new_key)
                        break
        if len(added_list) == 0:
            groups[k] = set()
            groups[k].update(cols)

    return groups
