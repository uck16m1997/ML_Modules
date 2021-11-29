from ml_package import *

# TODO Outlier Detection, Hyperparameter tuning

##### Initial Reading and Preprations #####
df = pd.read_table(
    "gpl_training _sample.tab.txt", delimiter="\t", encoding="ISO-8859-1"
)

# Remove Columns that contains no information
for c in df.columns:
    if len(df[c].unique()) == 1:
        df.drop(columns=[c], inplace=True)

# Column type inconsistencies
df["CUSTOMER_CITY"] = df["CUSTOMER_CITY"].astype(str)

df.loc[df["CUSTOMER_INCOME"] != "$null$", "CUSTOMER_INCOME"] = df[
    df["CUSTOMER_INCOME"] != "$null$"
]["CUSTOMER_INCOME"].astype(float)

df.loc[
    df["LAST_6_MNTH_TOTAL_NUM_LOANS"] != "$null$", "LAST_6_MNTH_TOTAL_NUM_LOANS"
] = df[df["LAST_6_MNTH_TOTAL_NUM_LOANS"] != "$null$"][
    "LAST_6_MNTH_TOTAL_NUM_LOANS"
].astype(
    int
)


# Replace missing values $null$ with na for practical reasons
df.replace(to_replace="$null$", value=np.nan, inplace=True)

## Convert string numeric columns
starts = ["LAS", "AVG", "MAX"]
num_convert = [c for c in df.select_dtypes(object).columns if c[:3] in starts]
df[num_convert] = df[num_convert].astype(float)

# CUSTOMER_ID and Musteri_No are the same and should not be a feature for classification
# REF_DATE is the same value as REFERENCE_DATE column in a different format
# CURRENT_CREDIT_LIMIT is the mostly similar to negated CUSTOMER_TOTAL_DEBT_AMOUNT and overlaps with TOTAL_OUTST_CREDIT_AMOUNT
# AVG_LAST_12_M_T_PAY_T_OUTS_BAL is same as other PAY_T_OUTS_BAL and same with LAST_MNTH_TOTAL_PAY_OUTS_BAL
# AVG_LAST_12_M_T_PAY_OUTS_BAL is being used for AVG_LAST_6_M_T_PAY_OUTS_BAL & AVG_LAST_3_M_T_PAY_OUTS_BAL
# MAX_DAYS_PASSED_DUE is the same as MAX_DAY_PASS_DUE_ALL_PRD_CUS
# LAST_12_MN_WORST_PAY_STATUS is being used for LAST_6_MN_WORST_PAY_STATUS
# CUSTOMER_BIRTH_DATE_YEAR is the same as AGE
# LAST_12_MNTH_MAX_DAY_PASS_DUE is being used for LAST_6_MNTH_MAX_DAY_PASS_DUE , LAST_3_MNTH_MAX_DAY_PASS_DUE and MAX_DAYS_PASSED_DUE,MAX_DAY_PASS_DUE_ALL_PRD_CUS
# LAST_12_MN_NUM_DELINQUENCY_1 is being used for LAST_3_MN_NUM_DELINQUENCY_1, LAST_6_MN_NUM_DELINQUENCY_1
# LAST_12_MN_NUM_DELINQUENCY_2 is being used for LAST_3_MN_NUM_DELINQUENCY_2, LAST_6_MN_NUM_DELINQUENCY_2
# LAST_12_MN_NUM_DELINQUENCY_3 is being used for LAST_3_MN_NUM_DELINQUENCY_3, LAST_6_MN_NUM_DELINQUENCY_3
# LAST_12_MN_NUM_DELINQUENCY_3_P  is being used for LAST_3_MN_NUM_DELINQUENCY_3_P, LAST_6_MN_NUM_DELINQUENCY_3_P
# LAST_12_MONTHS_TOTAL_PAYMENT is being used for  LAST_MONTH_TOTAL_PAYMENT, LAST_6_MNTH_TOTAL_PAYMENT, LAST_3_MONTHS_TOTAL_PAYMENT
# TIME_SINCE_FIRST_ACCOUNT_CR has high negative correlation with TIME_SINCE_CUSTOMER
# LAST_1_12_TOT_PAYMENT is being used for LAST_1_6_TOT_PAYMENT LAST_6_12_TOT_PAYMENT LAST_3_6_TOT_PAYMENT LAST_3_12_TOT_PAYMENT // LAST_1_3_TOT_PAYMENT
# LAST_6_MN_90_P_DPD_AMNT is being used for LAST_12_MN_30_P_DPD_AMNT // LAST_12_MN_60_P_DPD_AMNT
# PTT has high correlation for LAST_6_MNTH_TOTAL_NUM_LOANS
# INFM_FOR_RESTRUCTURING_YEAR -- INFM_FOR_RESTRUCTURING_MONTH INFM_FOR_RESTRUCTURING_DAY
df.drop(
    columns=[
        "CUSTOMER_ID",
        "MUSTERI_NO",
        "REF_DATE",
        # "CUSTOMER_TOTAL_DEBT_AMOUNT",
        # "TOTAL_OUTST_CREDIT_AMOUNT",
        # "LAST_MNTH_TOTAL_PAY_T_OUTS_BAL",
        # "AVG_LAST_6_M_T_PAY_T_OUTS_BAL",
        # "AVG_LAST_3_M_T_PAY_T_OUTS_BAL",
        # "LAST_MNTH_TOTAL_PAY_OUTS_BAL",
        # "AVG_LAST_6_M_T_PAY_OUTS_BAL",
        # "AVG_LAST_3_M_T_PAY_OUTS_BAL",
        # "LAST_6_MN_WORST_PAY_STATUS",
        "CUSTOMER_BIRTH_DATE",
        # "MAX_DAY_PASS_DUE_ALL_PRD_CUS",
        # "MAX_DAYS_PASSED_DUE",
        # "LAST_6_MNTH_MAX_DAY_PASS_DUE",
        # "LAST_3_MNTH_MAX_DAY_PASS_DUE",
        # "LAST_3_MN_NUM_DELINQUENCY_1",
        # "LAST_3_MN_NUM_DELINQUENCY_2",
        # "LAST_3_MN_NUM_DELINQUENCY_3",
        # "LAST_6_MN_NUM_DELINQUENCY_1",
        # "LAST_6_MN_NUM_DELINQUENCY_2",
        # "LAST_6_MN_NUM_DELINQUENCY_3",
        # "LAST_3_MN_NUM_DELINQUENCY_3_P",
        # "LAST_6_MN_NUM_DELINQUENCY_3_P",
        # "LAST_MONTH_TOTAL_PAYMENT",
        # "LAST_6_MNTH_TOTAL_PAYMENT",
        # "LAST_3_MONTHS_TOTAL_PAYMENT",
        # "TIME_SINCE_CUSTOMER",
        # "LAST_1_6_TOT_PAYMENT",
        # "LAST_6_12_TOT_PAYMENT",
        # "LAST_3_6_TOT_PAYMENT",
        # "LAST_3_12_TOT_PAYMENT",
        # "LAST_1_3_TOT_PAYMENT",
        "LAST_6_MN_WORST_STATUS",
        "MINI",
        "WEB",
        # "LAST_12_MN_30_P_DPD_AMNT",
        "LAST_6_MNTH_TOTAL_NUM_LOANS",
    ],
    inplace=True,
)

### Do Date Seperations

df = date_seperate(df)

###### Label feature processing ######
label_col = "NEW_DEFAULT_FLAG"

# Remove rows with missing labels if any
df.dropna(axis=0, subset=[label_col], inplace=True)

# Seperate Label from dataframe
y = df[label_col]
X = df.drop(columns=[label_col])
# Check Class Distribution: len(y[y == 1]) / len(y[y == 0])

###### Balance the data #####
# There are 1394 Defaults and 98606 Non Defaults in the Dataset
# Package: (pip install imbalanced-learn)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# over = SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=3)
# X, y = over.fit_resample(X, y)

# Oversample the minority class until it makes up 50% of the majority class
# over = RandomOverSampler(sampling_strategy=0.5, random_state=42)
# X, y = over.fit_resample(X, y)
# Check Class Distribution: len(y[y == 1]) / len(y[y == 0])

# # Undersample the majority class until minority class makes up 80% of the majority class
under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X, y = under.fit_resample(X, y)
# # Check Class Distribution: len(y[y == 1]) / len(y[y == 0])

# get schema for variables
Schema = X.dtypes
# get numeric and categoric columns
num_columns = X.select_dtypes(exclude=["object"]).columns
obj_columns = X.select_dtypes(include=["object"]).columns

from sklearn.impute import SimpleImputer

meanImputer = SimpleImputer(strategy="mean")
freqImputer = SimpleImputer(strategy="most_frequent")
##### !pip install category_encoders
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)

###Create the Model for classification
clf = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)


# Do kFold cross validation with 5 folds
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

from sklearn.preprocessing import PowerTransformer


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {"acc": [], "precision": [], "recall": []}
for train_index, test_index in kf.split(X, y):

    # Split the data in to test & train
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Do mean imputations for numeric columns
    X_train.loc[:, num_columns] = meanImputer.fit_transform(X_train[num_columns])
    X_test.loc[:, num_columns] = meanImputer.transform(X_test[num_columns])

    X_train.loc[:, obj_columns] = freqImputer.fit_transform(X_train[obj_columns])
    X_test.loc[:, obj_columns] = freqImputer.fit_transform(X_test[obj_columns])

    ## PCA Transformations will be done here
    X_train, X_test = PCATransformer(X_train, X_test, groups)
    # Numeric columns changed after pca
    num_columns_pca = X_train.select_dtypes(exclude=["object"]).columns

    encoder = ce.TargetEncoder(cols=obj_columns, return_df=True)

    X_train = encoder.fit_transform(X_train, y_train)
    X_test = encoder.transform(X_test)

    # scaler = StandardScaler()
    scaler = PowerTransformer(method="yeo-johnson")

    X_train[num_columns_pca] = scaler.fit_transform(X_train[num_columns_pca])
    X_test[num_columns_pca] = scaler.transform(X_test[num_columns_pca])

    # over = SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=3)
    # X_train, y_train = over.fit_resample(X_train, y_train)

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

for attributeIndex in range(0, X_train.shape[1]):
    print(results.pvalues[attributeIndex])
