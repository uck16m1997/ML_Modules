from ml_package import *
import dateutil


def date_seperate(X, y=None):
    X = X.copy()
    cols = X.select_dtypes(include=["object"]).columns
    for c in cols:
        try:
            if str(pd.to_datetime(X[c]).dtype) == "datetime64[ns]":
                date = pd.to_datetime(X[c])
                X[f"{c}_YEAR"] = date.dt.year
                X[f"{c}_MONTH"] = date.dt.month
                X[f"{c}_DAY"] = date.dt.day
                X.drop(columns=[c], inplace=True)
        except (dateutil.parser._parser.ParserError, ValueError):
            pass

    return X


def dis_col_ctrl(df, cols=None):
    dis_cols = []

    num_columns = cols if cols else df.columns
    for c in num_columns:
        try:
            if (df[~df[c].isna()][c].astype(int) == df[~df[c].isna()][c]).all():
                if len(df[c].unique()) / len(df[c]) <= 0.05:
                    dis_cols.append(c)
        except ValueError:
            pass
    return dis_cols


def cont_col_ctrl(df, cols=None):
    cont_cols = []
    obj_columns = cols if cols else df.columns
    for c in obj_columns:
        try:
            df[~df[c].isna()][c].astype(float)
            cont_cols.append(c)
        except ValueError:
            pass
    return cont_cols


def init_prep(df):
    ## Continious Numeric Columns
    cont_cols = cont_col_ctrl(df)
    df[cont_cols] = df[cont_cols].astype(float)
    ## Ordinal Categorical Columns
    dis_cols = dis_col_ctrl(df)

    cont_cols = pd.Index(cont_cols).difference(dis_cols)
    ## Seperate Dates into Year, Month, Day
    df = date_seperate(df)

    return (df, {"Continious": cont_cols, "Discrete": dis_cols})


def get_column_types(df):
    cont_cols = cont_col_ctrl(df)
    dis_cols = dis_col_ctrl(df)
    cont_cols = list(pd.Index(cont_cols).difference(dis_cols))
    cat_cols = list(df.columns.difference(dis_cols + list(cont_cols)))
    return {"Continious": cont_cols, "Discrete": dis_cols, "Categoric": cat_cols}


def find_inapp(df, null_thresh=0.7, many_vals=0.9, const_thresh=0.9):
    faulty_cols = {"Constant": [], "Unique": [], "Null": [], "Low Variance": []}
    for c in df.columns:
        # Remove Columns that contains no information
        if len(df[c].unique()) == 1:
            faulty_cols["Constant"].append(c)
        # Remove Categoric Columns that have too many unique values
        elif (
            str(df[c].dtype) == "object" and len(df[c].unique()) / len(df) >= many_vals
        ):
            faulty_cols["Unique"].append(c)
        # Remove columns with too much null
        elif df[c].isna().sum() / len(df) >= null_thresh:
            faulty_cols["Null"].append(c)
        # Remove columns with too litte variance
        elif max(df[c].value_counts()) / len(df) > const_thresh:
            faulty_cols["Low Variance"].append(c)
    return faulty_cols


def encode_cat(
    X_train,
    X_test,
    encoder,
    supervised=False,
    y_train=None,
    obj_columns=None,
    dimension_inc=False,
):
    if not obj_columns:
        obj_columns = X_train.select_dtypes(include=["object"]).columns

    if dimension_inc:
        return encode_inc_dim(X_train, X_test, encoder, obj_columns)

    X_train = X_train.copy()
    if X_test:
        X_test = X_test.copy()
        if supervised:
            X_train[obj_columns] = encoder.fit_transform(X_train[obj_columns], y_train)
            X_test[obj_columns] = encoder.transform(X_test[obj_columns])

        else:

            X_train[obj_columns] = encoder.fit_transform(X_train[obj_columns])
            X_test[obj_columns] = encoder.transform(X_test[obj_columns])
    else:
        if not supervised:
            X_train[obj_columns] = encoder.fit_transform(X_train[obj_columns])
        else:
            X_train[obj_columns] = encoder.fit_transform(X_train[obj_columns], y_train)

    return X_train, X_test


def encode_inc_dim(X_train, X_test, encoder, obj_columns=None, const_thresh=0.9):

    X_train = X_train.copy()

    tmp_encoded = encoder.fit_transform(X_train[obj_columns])
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


# X =pd.Series([53, 56, 57, 63, 66, 67, 67,67, 68, 69, 70, 70, 70, 70, 72, 73, 75, 75,
#        76, 76, 78, 79, 80, 81])
# y =pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
#        0, 0])


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


# def find_susp(df):
#     # vif_info = pd.DataFrame()
#     # vif_info["VIF"] = [
#     #     variance_inflation_factor(X.values, i) for i in range(X.shape[1])
#     # ]
#     # vif_info["Column"] = X.columns
#     # vif_info.sort_values("VIF", ascending=False)
#     # vif_info[vif_info["VIF"].values > 5]
