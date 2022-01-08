from ml_package import *
import eda

# Set the parameters for the Initial Prep Functions

folder = "Titanic/"
train_file = "train.csv"
target = "Survived"
test_file = "test.csv"
method = "Classification"


def create_json_conf(d, f_name):
    with open("Config/" + f_name, "w") as outfile:
        json.dump(d, outfile)


if __name__ == "__main__":

    # Read the data
    X_train = pd.read_csv(folder + train_file)

    # Fix Null Values
    X_train = X_train.replace("$null$", np.nan)
    # Remove rows with missing labels if any_train
    X_train.dropna(axis=0, subset=[target], inplace=True)

    # Seperate Target from Data
    y_train = X_train[target]
    X_train.drop(columns=[target], inplace=True)

    # Find inappropriate features
    col_dict = data_prep.find_inapp(X_train)
    # Remove constant features
    X_train.drop(columns=col_dict["Constant"], inplace=True)

    # Remove columns with too many_train unique values
    X_train.drop(columns=col_dict["Unique"], inplace=True)

    # Remove columns with too many_train null values
    X_train.drop(columns=col_dict["Null"], inplace=True)

    # Remove categoric columns with too little variance
    X_train.drop(columns=col_dict["Low Variance"], inplace=True)

    X_train, col_details = data_prep.init_prep(X_train)

    # Take a look at the discrete variables
    X_train[col_details["Continious"]]

    # Take a look at the discrete variables
    X_train[col_details["Discrete"]]

    # Take a look at the discrete variables
    X_train[col_details["Categoric"]]

    ## MANUAL HERE ##
    ## FIX INCONSISTENCIES ##

    # Get Final Column Details and Create Config file
    col_details = data_prep.get_column_types(X_train)
    create_json_conf(col_details, "ColumnConf.json")

    ## if classification asses the need for sampling methods
    class_sample = False
    if method == "Classification":
        dist_thresh = 1 / (len(y_train.unique()) * 2)
        min_class = min(y_train.value_counts())
        if min_class / len(y_train) < 1 / (len(y_train.unique()) * 2):
            class_sample = True

    # Get Correlation Groups
    groups = custom_comps.CorrelationClustering(
        X_train,
    )
    # will there be groups dimensionality_train reduction
    group_funcs = False
    if groups:
        for g in groups:
            groups[g] = list(groups[g])
        create_json_conf(groups, "GroupsConf.json")
        group_funcs = True

    unique_per = 2
    high_cardinality = []
    # Categoric Columns with high cardinality_train
    for c in col_details["Categoric"]:
        if len(X_train[c].unique()) > round(len(X_train) ** (1 / unique_per)):
            high_cardinality.append(c)

    # Create Run Configs
    run_config = {
        "class_sampling": class_sample,
        "group_funcs": group_funcs,
        "method": method,
        "high_cardinality": high_cardinality,
    }
    create_json_conf(run_config, "RunConf.json")

    # If there is a test file
    if test_file:
        X_test = pd.read_csv(folder + test_file)
        # Fix types for the test sets
        X_test = X_test.astype(X_train.dtypes)
        # Remove constant features
        X_test.drop(columns=col_dict["Constant"], inplace=True)
        # Remove categoric columns with too many_train unique values
        X_test.drop(columns=col_dict["Unique"], inplace=True)
        # Remove columns with too many_train null values
        X_test.drop(columns=col_dict["Null"], inplace=True)
        # Remove categoric columns with too little variance
        X_test.drop(columns=col_dict["Low Variance"], inplace=True)
        X_test.to_parquet("Data/test.parquet")

    X_train["Target"] = y_train
    X_train.to_parquet("Data/train.parquet")
