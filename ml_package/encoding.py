from ml_package import *


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
    if drop_first:
        for c in obj_columns:

            # Encode obj_columns
            encoder.cols = [c]
            encoder.fit(X_train[c])
            tmp_encoded = encoder.transform(X_train[c])
            # Drop original columns
            X_train.drop(columns=c, inplace=True)
            X_train[tmp_encoded.columns[:-1]] = tmp_encoded[tmp_encoded.columns[:-1]]
            if X_test:
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
        if X_test:
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
