from ml_package import *
import eda

# Set the parameters for the Initial Prep Functions

folder = "titanic/"
target = "Survived"
method = "Classification"


def create_json_conf(d, fName):
    with open("Config/" + fName, "w") as outfile:
        json.dump(d, outfile)


if __name__ == "__main__":

    # Read the data
    X_train = pd.read_csv(folder + "train.csv")
    X_test = pd.read_csv(folder + "test.csv")
    # Remove rows with missing labels if any_train
    X_train.dropna(axis=0, subset=[target], inplace=True)

    # Seperate Target from Data
    y_train = X_train[target]
    X_train.drop(columns=[target], inplace=True)

    X_train, col_details = data_prep.init_prep(X_train)

    # Take a look at the discrete variables
    X_train[col_details["Continious"]]

    # Take a look at the discrete variables
    X_train[col_details["Discrete"]]

    # Take a look at the discrete variables
    X_train[col_details["Categoric"]]

    # Fix types for the test sets
    X_test = X_test.astype(X_train.dtypes)
    ## MANUAL HERE ##
    ## FIX INCONSISTENCIES ##
    ## DO SPECIFIC OPERATIONS ##

    # Find inappropriate features
    col_dict = data_prep.find_inapp(X_train)
    # Remove constant features
    X_train.drop(columns=col_dict["Constant"], inplace=True)
    X_test.drop(columns=col_dict["Constant"], inplace=True)

    # Remove categoric columns with too many_train unique values
    X_train.drop(columns=col_dict["Unique"], inplace=True)
    X_test.drop(columns=col_dict["Unique"], inplace=True)

    # Remove categoric columns with too many_train null values
    X_train.drop(columns=col_dict["Null"], inplace=True)
    X_test.drop(columns=col_dict["Null"], inplace=True)

    # Remove categoric columns with too little variance
    X_train.drop(columns=col_dict["Low Variance"], inplace=True)
    X_test.drop(columns=col_dict["Low Variance"], inplace=True)

    # Get Final Column Details and Create Config file
    col_details = data_prep.get_column_types(X_train)
    create_json_conf(col_details, "ColumnConf.json")

    ## if classification asses the need for sampling methods
    class_sample = False
    if method == "Classification":
        dist_thresh = 1 / (len(y_train.unique()) * 2)
        min_class = min(y_train.value_counts())
        if min_class / len(y_train) < 0.3:
            class_sample = True

    # Get Correlation Groups
    groups = custom_comps.CorrelationClustering(
        X_train,
    )
    # will there be groups dimensionality_train reduction
    group_funcs = False
    if groups:
        create_json_conf(col_details, "GroupsConf.json")
        group_funcs = True

    unique_per = 64
    high_cardinality = []
    # Categoric Columns with high cardinality_train
    for c in col_details["Categoric"]:
        if len(X_train[c].unique()) > len(X_train) ** 1 / unique_per:
            high_cardinality.append(c)

    # Create Run Configs
    run_config = {
        "class_sampling": class_sample,
        "group_funcs": group_funcs,
        "method": method,
        "high_cardinality": high_cardinality,
    }
    create_json_conf(run_config, "RunConf.json")

    X_train["Target"] = y_train
    X_train.to_parquet("Data/train.parquet")

    X_test.to_parquet("Data/test.parquet")
