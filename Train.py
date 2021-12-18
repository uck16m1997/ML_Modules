from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from ml_package import *

### Load the Data needed for the process ###
proc = process.init_import()
X, y = proc["Train"]
col_details, run_details = proc["Configs"]

###Create the Model for classification
clf = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
## Create the kfold instance
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metrics = {"acc": [], "precision": [], "recall": []}

train_flow = []
with open("Config/TrainConf.pickle", "rb") as f:
    train_flow = pickle.load(f)

for train_index, test_index in kf.split(X, y):

    # Split the data in to test & train
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # If there is a need to data sample
    if run_details["class_sampling"]:
        X_train, y_train = data_sampling.under_sample(X_train, y_train, 0.5, 42)

    X_train, X_test = process.train_pipeline(X_train, y_train, X_test, train_flow)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    metrics["acc"].append(accuracy_score(y_test, pred))
    metrics["precision"].append(precision_score(y_test, pred))
    metrics["recall"].append(recall_score(y_test, pred))


# import statsmodels.api as sm

# X_train["INTERCEPT"] = 1
# # X_train.drop(columns=["Intercept"], inplace=True)
# model = sm.Logit(y_train, X_train)
# results = model.fit(method="bfgs", maxiter=1000)
# results.summary()

# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, pred)
# cm
