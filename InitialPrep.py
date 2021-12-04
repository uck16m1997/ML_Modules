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
    df = pd.read_csv(folder + "train.csv")

    # Remove rows with missing labels if any
    df.dropna(axis=0, subset=[target], inplace=True)

    # Seperate Target from Data
    y = df[target]
    df.drop(columns=[target], inplace=True)

    df, col_details = data_prep.init_prep(df)

    # Take a look at the discrete variables
    df[col_details["Continious"]]

    # Take a look at the discrete variables
    df[col_details["Discrete"]]

    # Take a look at the discrete variables
    df[col_details["Categoric"]]

    ## MANUAL HERE ##
    ## FIX INCONSISTENCIES ##
    ## DO SPECIFIC OPERATIONS ##

    # Find inappropriate features
    col_dict = data_prep.find_inapp(df)
    # Remove constant features
    df.drop(columns=col_dict["Constant"], inplace=True)
    # Remove categoric columns with too many unique values
    df.drop(columns=col_dict["Unique"], inplace=True)
    # Remove categoric columns with too many null values
    df.drop(columns=col_dict["Null"], inplace=True)
    # Remove categoric columns with too little variance
    df.drop(columns=col_dict["Low Variance"], inplace=True)

    # Get Final Column Details and Create Config file
    col_details = data_prep.get_column_types(df)
    create_json_conf(col_details, "ColumnConf.json")

    ## if classification asses the need for sampling methods
    class_sample = False
    if method == "Classification":
        min_class = min(y.value_counts())
        if min_class / len(y) < 0.3:
            class_sample = True

    # Get Correlation Groups
    groups = custom_comps.CorrelationClustering(
        df,
    )
    # will there be groups dimensionality reduction
    group_funcs = False
    if groups:
        create_json_conf(col_details, "GroupsConf.json")
        group_funcs = True

    # Create Run Configs
    run_config = {
        "class_sampling": class_sample,
        "group_funcs": group_funcs,
        "method": method,
    }
    create_json_conf(run_config, "RunConf.json")

    df["Target"] = y
    df.to_parquet("Data/train.parquet")
