from ml_package import *
from sklearn.decomposition import PCA
import math


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
    distance_mat.replace({0: 1}, inplace=True)
    values = np.sort((np.unique(distance_mat.values)))
    values = values[values < threshold]

    for v in values:
        cols = list(distance_mat.iloc[np.where(distance_mat == v)].columns)
        k = "_".join(cols)
        added_list = []
        keys = list(groups.keys())
        for key in keys:
            for c in cols:
                if c in key:
                    if len(added_list):
                        groups[key].update(groups[added_list[0]])
                        new_key = "_".join(groups[key])
                        groups[new_key] = groups.pop(key)
                        groups.pop(added_list[0])
                        added_list = [new_key]
                    else:
                        groups[key].update(cols)
                        new_key = "_".join(groups[key])
                        groups[new_key] = groups.pop(key)
                        added_list.append(new_key)

        if len(added_list) == 0:
            groups[k] = set()
            groups[k].update(cols)

    return groups
