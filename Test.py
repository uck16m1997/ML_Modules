from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from ml_package import *

### Load the Data that is outputted by Initial Prep ###
proc = process.init_import("Test")
X_train, y_train = proc["Train"]
X_test = proc["Test"]
col_details, run_details = proc["Configs"]

# If there is a need to data sample
if run_details["class_sampling"]:
    X_train, y_train = data_sampling.under_sample(X_train, y_train, 0.5, 42)

train_flow = []
with open("Config/TrainConf.pickle", "rb") as f:
    train_flow = pickle.load(f)

X_train, X_test = process.train_pipeline(X_train, y_train, X_test, train_flow)

###Create the Model for classification
clf = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

pd.DataFrame(
    {"PassengerId": pd.read_csv("Titanic/test.csv")["PassengerId"], "Survived": pred}
).to_csv("Predictions.csv", index=False)
