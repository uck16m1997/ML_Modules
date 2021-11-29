from eda import groups, eda_prep
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from ml_package import *

### Load the Data with insights and fixes learned from eda process ###
X, y = eda_prep()

# Find inappropriate features
col_dict = data_prep.find_inapp(X)
# Remove constant features
X.drop(columns=col_dict["Constant"], inplace=True)
# Remove categoric columns with too many unique values
X.drop(columns=col_dict["Unique"], inplace=True)
# Remove categoric columns with too many null values
X.drop(columns=col_dict["Null"], inplace=True)
# Remove categoric columns with too little variance
X.drop(columns=col_dict["Low Variance"], inplace=True)

# Find correlations among data
### Not Automated done by groups in EDA ###
### To Do Automate Correlation Groups From Correlation Matrix ###
### Control the Variance Inflation Factor and AutoGroup PC_i P values ###


# Under Sampling for more balanced target distribution
from imblearn.under_sampling import RandomUnderSampler

under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X, y = under.fit_resample(X, y)

cat_cols = X.select_dtypes(include = "object").columns
# Create Dummy Variables here
encoder = ce.OneHotEncoder()
X, tmp, low_vars, encoded_cols = data_prep.encode_cat(
    X,
    X_test=None,
    encoder=encoder,
    dimension_inc=True,
)
X.drop(columns=low_vars, inplace=True)

###Create the Model for classification
clf = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {"acc": [], "precision": [], "recall": []}

col_details = data_prep.get_column_types(X)


for train_index, test_index in kf.split(X, y):

    # Split the data in to test & train
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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
    bin_cols = ["AGE"]
    X_train[bin_cols], X_test[bin_cols] = custom_comps.InfoGainDiscretizer(
        X_train[bin_cols], X_test[bin_cols], y_train
    )

    ## PCA Transformation
    X_train, X_test = custom_comps.PCATransformer(X_train, X_test, groups)

    # Encode categorical data
    # encoder = ce.TargetEncoder()
    # data_prep.encode_cat(X_train, X_test, encoder, supervised=True, y_train=y_train)

    # Center Scale PowerTransform the Data
    scaler = RobustScaler()
    pt = PowerTransformer(method="yeo-johnson")

    X_train, X_test = custom_comps.CenterScaleTransform(X_train, X_test, scaler, pt)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    metrics["acc"].append(accuracy_score(y_test, pred))
    metrics["precision"].append(precision_score(y_test, pred))
    metrics["recall"].append(recall_score(y_test, pred))


import statsmodels.api as sm

X_train["INTERCEPT"] = 1
# X_train.drop(columns=["Intercept"], inplace=True)
model = sm.Logit(y_train, X_train)
results = model.fit(method="bfgs", maxiter=1000)
results.summary()

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
cm
