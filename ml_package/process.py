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


def train_pipeline(
    X_train,
    y_train,
    X_test,
    pipeline,
):
    for i, p in enumerate(pipeline):
        print(f"Started Step_{i+1}: {p[0]} ")
        args = p[1]
        transformer = args["transformer"]
        fit_params = [X_train]
        fit_params.extend(args["fit"]["Input"])
        if "Append" in args["fit"].keys():
            for a in args["fit"]["Append"]:
                if a == "y":
                    fit_params.append(y_train)
                elif a == "noncont":
                    col_details = data_prep.get_column_types(X_train)
                    fit_params.append(
                        col_details["Discrete"] + col_details["Categoric"]
                    )

        X_train = transformer.fit_transform(*fit_params)
        X_test = transformer.transform(X_test)
    return X_train, X_test
