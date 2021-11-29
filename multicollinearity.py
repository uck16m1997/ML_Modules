from eda import groups, eda_prep
from ml_package import *


if __name__ == "__main__":
    ##### Load the Data with insights and fixes learned from eda process #####
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

    from sklearn.impute import SimpleImputer

    meanImputer = SimpleImputer(strategy="mean")
    freqImputer = SimpleImputer(strategy="most_frequent")

    # get numeric and categoric columns
    num_columns = X.select_dtypes(exclude=["object"]).columns
    obj_columns = X.select_dtypes(include=["object"]).columns

    import category_encoders as ce
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer

    X.loc[:, num_columns] = meanImputer.fit_transform(X[num_columns])

    X.loc[:, obj_columns] = freqImputer.fit_transform(X[obj_columns])

    encoder = ce.TargetEncoder(cols=obj_columns, return_df=True)

    X = encoder.fit_transform(X, y)

    scaler = StandardScaler()

    X[num_columns] = scaler.fit_transform(X[num_columns])

    X = custom_comps.PCATransformer(X, X, groups)[0]

    X["Intercept"] = 1
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_info = pd.DataFrame()
    vif_info["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    vif_info["Column"] = X.columns
    vif_info.sort_values("VIF", ascending=False)
    vif_info[vif_info["VIF"].values > 5]

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.clf()
    plt.figure(figsize=(18, 18), dpi=80)
    # [vif_info[vif_info["VIF"].values > 5]["Column"]]
    cor_mat = X.corr()
    sns.heatmap(cor_mat)

    X["TIME_SINCE_CUSTOMER"]
    X["REFERENCE_DATE_YEAR"]
    X["TIME_SINCE_FIRST_ACCOUNT_CR_YEAR"]

    model = sm.Logit(y, X)
    results = model.fit(method="newton", maxiter=1000)
    results.summary()
