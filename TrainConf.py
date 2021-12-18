from ml_package import *

### Load the Data needed for the process ###
proc = process.init_import()
X, y = proc["Train"]
col_details, run_details = proc["Configs"]

# Optional add columns you wanted binned
bin_cols = ["Parch", "SibSp"]


# Create the train flow
train_flow = []


# Add Encoding step for the train flow
train_flow.append(
    (
        "Encoding",
        {
            "transformer": encoding.Upper_Dimension_Encoder(),
            "fit": {
                "Input": [
                    ce.OneHotEncoder(use_cat_names=True),
                    pd.Index(col_details["Categoric"]).difference(
                        run_details["high_cardinality"]
                    ),
                ]
            },
        },
    )
)
# Add imputation step
train_flow.append(
    (
        "Imputation",
        {
            "transformer": imputation.Impute_Transformer(),
            "fit": {
                "Input": [
                    SimpleImputer(strategy="mean"),
                    SimpleImputer(strategy="most_frequent"),
                    col_details["Continious"],
                ],
                "Append": ["noncont"],
            },
        },
    )
)


# if bin columns are determined add binning
if len(bin_cols) > 0:
    train_flow.append(
        (
            "Binning",
            {
                "transformer": binning.Info_Gain_Discretizer(),
                "fit": {
                    "Input": [bin_cols],
                    "Append": ["y"],
                },
            },
        )
    )


# Add encoding for Categoric Columns with High Cardinality
train_flow.append(
    (
        "High_Cardinality_Encoding",
        {
            "transformer": encoding.Categorical_Encoder(supervised=True),
            "fit": {
                "Input": [
                    ce.LeaveOneOutEncoder(),
                    run_details["high_cardinality"],
                ],
                "Append": ["y"],
            },
        },
    )
)

# Add Centering step for the data
train_flow.append(
    (
        "Centering",
        {
            "transformer": custom_comps.Center_Transformer(),
            "fit": {
                "Input": [
                    col_details["Continious"],
                ],
            },
        },
    )
)


# Add Scaling step for the data
train_flow.append(
    (
        "Scaling",
        {
            "transformer": custom_comps.Scaling_Transformer(),
            "fit": {
                "Input": [
                    RobustScaler(),
                    col_details["Continious"],
                ],
            },
        },
    )
)


# if needed add dimensionality reduction
if run_details["group_funcs"]:
    train_flow.append(
        (
            "Dimensionality Reduction",
            {
                "transformer": custom_comps.PCA_Transformer(),
                "fit": {
                    "Input": [
                        proc["groups"],
                    ],
                },
            },
        )
    )

with open("Config/TrainConf.pickle", "wb") as file:
    pickle.dump(train_flow, file)
