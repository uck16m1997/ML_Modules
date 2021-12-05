from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from ml_package import *

### Load the Data that is outputted by Initial Prep ###
proc = process.init_import("Test")
X_train, y_train = proc["Train"]
X_test = proc["Test"]
col_details, run_details = proc["Configs"]

# If there is a need to data sample
if run_details["class_sampling"]:
    X_train, y_train = data_sampling.under_sample(X_train, y_train, 0.5, 42)

# If there are high cardinality columns
cat_cols = X_train.select_dtypes(include="object").columns
cat_cols = cat_cols.difference(run_details["high_cardinality"])
# Create Dummy Variables here
encoder = ce.OneHotEncoder(use_cat_names=True)
X_train, X_test, low_vars, encoded_cols = encoding.encode_inc_dim(
    X_train, X_test=X_test, encoder=encoder, obj_columns=cat_cols
)
X_train.drop(columns=low_vars, inplace=True)
X_test.drop(columns=low_vars, inplace=True)

# Get Column Details
col_details = data_prep.get_column_types(X_train)

###Create the Model for classification
clf = GradientBoostingClassifier(random_state=42)

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
    X_train, X_test = custom_comps.PCATransformer(X_train, X_test, groups)

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

pd.DataFrame(
    {"PassengerId": pd.read_csv("Titanic/test.csv")["PassengerId"], "Survived": pred}
).to_csv("Predictions.csv", index=False)
