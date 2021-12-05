from ml_package import *


def init_import(flow="Train"):
    # Initizialize Proc dict
    proc = {}

    # Load the Data that is outputted by Initial Prep
    X = pd.read_parquet("Data/train.parquet")
    y = X["Target"]
    X.drop(columns=["Target"], inplace=True)

    if flow == "Test":
        X_test = pd.read_parquet("Data/test.parquet")
        proc["Test"] = X_test

    proc["Train"] = (X, y)

    ### Load the Config files
    with open("Config/ColumnConf.json") as f:
        col_details = json.load(f)
    with open("Config/RunConf.json") as f:
        run_details = json.load(f)
    if run_details["group_funcs"]:
        with open("Config/GroupsConf.json") as f:
            proc["groups"] = json.load(f)
    proc["Configs"] = (col_details, run_details)

    return proc
