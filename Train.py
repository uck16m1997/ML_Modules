from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from ml_package import *

### Load the Data needed for the process ###
proc = process.init_import()
X, y = proc["Train"]
col_details, run_details = proc["Configs"]

###Create the Model for classification
clf = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
## Create the kfold instance
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metrics = {"acc": [], "precision": [], "recall": []}

for train_index, test_index in kf.split(X, y):

    # Split the data in to test & train
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # If there is a need to data sample
    if run_details["class_sampling"]:
        X_train, y_train = data_sampling.under_sample(X_train, y_train, 0.5, 42)

    # If there are high cardinality columns
    cat_cols = X.select_dtypes(include="object").columns
    cat_cols = cat_cols.difference(run_details["high_cardinality"])

    # Create Dummy Variables here
    encoder = ce.OneHotEncoder(use_cat_names=True)
    a = encoding.encode_inc_dim(
        X_train, X_test=X_test, encoder=encoder, obj_columns=cat_cols
    )
    # X_train, X_test, low_vars, encoded_cols = a
    X_train = a[0]
    X_test = a[1]
    low_vars = a[2]
    X_train.drop(columns=low_vars, inplace=True)
    X_test.drop(columns=low_vars, inplace=True)

    col_details = data_prep.get_column_types(X_train)

    # Impute missing data
    meanImputer = SimpleImputer(strategy="mean")
    freqImputer = SimpleImputer(strategy="most_frequent")

    X_train, X_test = data_prep.impute_missing(
        X_train,
        X_test,
        numImputer=meanImputer,
        catImputer=freqImputer,
        num_columns=col_details["Continious"],
        obj_columns=col_details["Discrete"] + col_details["Categoric"],
    )

    # Information Gain Binning
    # bin_cols = ["AGE"]
    # X_train[bin_cols], X_test[bin_cols] = custom_comps.InfoGainDiscretizer(
    #     X_train[bin_cols], X_test[bin_cols], y_train
    # )

    ## Dimensionality Reduction
    if run_details["group_funcs"]:
        X_train, X_test = custom_comps.PCATransformer(X_train, X_test, proc["groups"])

    # Encode categorical data
    encoder = ce.LeaveOneOutEncoder()
    (X_train, X_test) = encoding.encode_cat(
        X_train,
        X_test,
        encoder,
        supervised=True,
        y_train=y_train,
        obj_columns=run_details["high_cardinality"],
    )

    # Center Scale PowerTransform the Data
    scaler = RobustScaler()
    pt = PowerTransformer(method="yeo-johnson")

    X_train, X_test = custom_comps.CenterScaleTransform(X_train, X_test, scaler, pt)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    metrics["acc"].append(accuracy_score(y_test, pred))
    metrics["precision"].append(precision_score(y_test, pred))
    metrics["recall"].append(recall_score(y_test, pred))


# import statsmodels.api as sm

# X_train["INTERCEPT"] = 1
# # X_train.drop(columns=["Intercept"], inplace=True)
# model = sm.Logit(y_train, X_train)
# results = model.fit(method="bfgs", maxiter=1000)
# results.summary()

# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, pred)
# cm
