from ml_package import *


class Categorical_Encoder:
    def __init__(self, supervised=False):
        self.supervised = supervised

    def fit(self, X_train, encoder, obj_columns=[], y_train=None):
        if len(obj_columns) == 0:
            self.obj_columns = X_train.select_dtypes(include=["object"]).columns
        else:
            self.obj_columns = obj_columns

        self.encoder = encoder

        if self.supervised:
            self.encoder.fit(X_train[self.obj_columns], y_train)
        else:
            self.encoder.fit(X_train[self.obj_columns])

    def transform(self, X):
        X.loc[:, self.obj_columns] = self.encoder.transform(X.loc[:, self.obj_columns])
        return X

    def fit_transform(self, X_train, encoder, obj_columns=[], y_train=None):
        self.fit(X_train, encoder, obj_columns, y_train)
        return self.transform(X_train)


class Upper_Dimension_Encoder:
    def __init__(self, drop_first=True):
        self.drop_first = drop_first

    def fit(self, X_train, encoder, obj_columns=[]):
        if len(obj_columns) == 0:
            self.obj_columns = X_train.select_dtypes(include=["object"]).columns
        else:
            self.obj_columns = obj_columns

        self.encoder = encoder
        self.encoder.cols = obj_columns

        self.encoder.fit(X_train[self.obj_columns])

    def transform(self, X, drop_invariant=1):
        res = self.encoder.transform(X.loc[:, self.obj_columns])
        if self.drop_first:
            remove_list = list(self.encoder.cols[:])
            for c in self.encoder.feature_names:
                if c.split("_")[0] in remove_list:
                    res.drop(columns=[c], inplace=True)
                    remove_list.remove(c.split("_")[0])
        X = X.drop(columns=self.obj_columns)
        X[res.columns] = res

        if drop_invariant < 1:
            faulty_cols = []
            for c in X_train.columns:
                if max(X_train[c].value_counts()) / len(X_train) > drop_invariant:
                    faulty_cols.append(c)
            self.faulty_cols = faulty_cols
            X.drop(columns=faulty_cols, inplace=True)

        return X

    def fit_transform(self, X_train, encoder, obj_columns=[], drop_invariant=1):
        self.fit(X_train, encoder, obj_columns)
        return self.transform(X_train, drop_invariant)


def encode_cat(
    X_train,
    X_test,
    encoder,
    supervised=False,
    y_train=None,
    obj_columns=None,
):
    try:
        if not obj_columns:
            obj_columns = X_train.select_dtypes(include=["object"]).columns
    except ValueError:
        pass

    X_train = X_train.copy()
    try:
        if not X_test:
            if not supervised:
                X_train[obj_columns] = encoder.fit_transform(X_train[obj_columns])
            else:
                X_train[obj_columns] = encoder.fit_transform(
                    X_train[obj_columns], y_train
                )

    except ValueError:
        # Throws ValueError when X_test is a DataFrame

        X_test = X_test.copy()
        if supervised:
            X_train[obj_columns] = encoder.fit_transform(X_train[obj_columns], y_train)
            X_test[obj_columns] = encoder.transform(X_test[obj_columns])

        else:

            X_train[obj_columns] = encoder.fit_transform(X_train[obj_columns])
            X_test[obj_columns] = encoder.transform(X_test[obj_columns])

    return X_train, X_test


def encode_inc_dim(
    X_train, X_test, encoder, obj_columns=None, const_thresh=0.9, drop_first=True
):

    try:
        if not obj_columns:
            obj_columns = X_train.select_dtypes(include=["object"]).columns
    except ValueError:
        pass

    # Copy train
    X_train = X_train.copy()
    tmp_encoded = encoder.fit_transform(X_train[obj_columns])
    if drop_first:
        for c in obj_columns:

            # Encode obj_columns
            encoder.cols = [c]
            encoder.fit(X_train[c])
            tmp_encoded = encoder.transform(X_train[c])
            # Drop original columns
            X_train.drop(columns=c, inplace=True)
            X_train[tmp_encoded.columns[:-1]] = tmp_encoded[tmp_encoded.columns[:-1]]
            try:
                if X_test == None:
                    pass
            except:
                # Encode obj_columns
                tmp_encoded = encoder.transform(X_test[c])
                # Drop original columns
                X_test.drop(columns=c, inplace=True)
                X_test[tmp_encoded.columns[:-1]] = tmp_encoded[tmp_encoded.columns[:-1]]
    else:
        # Encode obj_columns
        tmp_encoded = encoder.fit_transform(X_train[obj_columns])
        # Drop original columns
        X_train.drop(columns=obj_columns, inplace=True)
        X_train[tmp_encoded.columns] = tmp_encoded
        try:
            if X_test == None:
                pass
        except:
            X_test = X_test.copy()
            tmp_encoded = encoder.transform(X_test[obj_columns])
            X_test.drop(columns=obj_columns, inplace=True)
            X_test[tmp_encoded.columns] = tmp_encoded

    faulty_cols = []
    for c in X_train.columns:
        if max(X_train[c].value_counts()) / len(X_train) > const_thresh:
            faulty_cols.append(c)

    return X_train, X_test, faulty_cols, tmp_encoded.columns


def cat_embeddings(X, y, cat_cols):
    for cat_col in cat_cols:
        col_encodes = []
        for c in X.columns:
            if cat_col in c:
                col_encodes.append(c)

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(
                int(math.sqrt(len(col_encodes))), input_shape=(len(col_encodes),)
            )
        )
        model.add(
            tf.keras.layers.Dense(1, input_shape=(int(math.sqrt(len(col_encodes))),))
        )
        # model.fit()
