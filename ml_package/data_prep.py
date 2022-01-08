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
        except (dateutil.parser._parser.ParserError, ValueError, TypeError):
            pass

    return X


def dis_col_ctrl(df, cols=None):
    dis_cols = []
    num_columns = cols if cols else df.columns
    for c in num_columns:
        try:
            if (df[~df[c].isna()][c].astype(int) == df[~df[c].isna()][c]).all():
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

    ## Seperate Dates into Year, Month, Day
    df = date_seperate(df)
    ## Continious Numeric Columns
    cont_cols = cont_col_ctrl(df)
    df[cont_cols] = df[cont_cols].astype(float)
    ## Ordinal Categorical Columns
    dis_cols = dis_col_ctrl(df)
    cont_cols = list(pd.Index(cont_cols).difference(dis_cols))

    cat_cols = list(df.columns.difference(dis_cols + cont_cols))

    return (df, {"Continious": cont_cols, "Discrete": dis_cols, "Categoric": cat_cols})


def get_column_types(df):
    cont_cols = cont_col_ctrl(df)
    dis_cols = dis_col_ctrl(df)
    cont_cols = list(pd.Index(cont_cols).difference(dis_cols))
    cat_cols = list(df.columns.difference(dis_cols + list(cont_cols)))
    return {"Continious": cont_cols, "Discrete": dis_cols, "Categoric": cat_cols}


def find_inapp(df, null_thresh=0.5, unique_thresh=0.8, const_thresh=0.9):
    faulty_cols = {"Constant": [], "Unique": [], "Null": [], "Low Variance": []}
    for c in df.columns:
        # Remove Columns that contains no information
        if len(df[c].unique()) == 1:
            faulty_cols["Constant"].append(c)
        # Remove columns with too much null
        elif df[c].isna().sum() / len(df) >= null_thresh:
            faulty_cols["Null"].append(c)
        # Remove columns with too litte variance
        elif max(df[c].value_counts()) / len(df) > const_thresh:
            faulty_cols["Low Variance"].append(c)
        # Remove non Continious Columns that have too many unique values
        else:
            # If Categorical
            not_cont = False
            if str(df[c].dtype) == "object":
                try:
                    df[c].astype(float)
                except ValueError:
                    not_cont = True

            # If discrete
            if not not_cont:
                try:
                    if (df[~df[c].isna()][c].astype(int) == df[~df[c].isna()][c]).all():
                        not_cont = True
                except ValueError:
                    pass
            # If confirmed non continious
            if not_cont and len(df[c].unique()) / len(df) >= unique_thresh:
                faulty_cols["Unique"].append(c)

    return faulty_cols


# X =pd.Series([53, 56, 57, 63, 66, 67, 67,67, 68, 69, 70, 70, 70, 70, 72, 73, 75, 75,
#        76, 76, 78, 79, 80, 81])
# y =pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
#        0, 0])


# def find_susp(df):
#     # vif_info = pd.DataFrame()
#     # vif_info["VIF"] = [
#     #     variance_inflation_factor(X.values, i) for i in range(X.shape[1])
#     # ]
#     # vif_info["Column"] = X.columns
#     # vif_info.sort_values("VIF", ascending=False)
#     # vif_info[vif_info["VIF"].values > 5]
