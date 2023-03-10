
# wrapper for logistic regression
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from fairxplainer import utils
import os
import pickle
import random



def init(dataset, classifier="lr", repaired=False, verbose=False, compute_equalized_odds=False, remove_column=None, fraction=0.5):
    df = dataset.get_df(repaired=repaired)

    random.seed(10)

    # discretize
    # df =  utils.get_discretized_df(df, columns_to_discretize=dataset.continuous_attributes)

    # get X,y
    X = df.drop(['target'], axis=1)
    y = df['target']

    if(remove_column is not None):
        assert isinstance(remove_column, str)
        X = X.drop([remove_column], axis=1)

    if(fraction < 1):
        raise NotImplementedError()
        """Not correctly implemented. Cannot handle cases where sensitive features are non-binary but categorical. 
           Also, random shuffle is required. 
        """
        non_sensitive_attributes = [
            attribute for attribute in X.columns if attribute not in dataset.known_sensitive_attributes]
        sub_columns = random.sample(non_sensitive_attributes, int(
            fraction * len(non_sensitive_attributes)))
        X = X[sub_columns + dataset.known_sensitive_attributes]

    # one-hot
    X = utils.get_one_hot_encoded_df(X, dataset.categorical_attributes)
    # X = utils.get_one_hot_encoded_df(X,X.columns.to_list())

    skf = KFold(n_splits=5, shuffle=True, random_state=10)
    skf.get_n_splits(X, y)

    X_trains = []
    y_trains = []
    X_tests = []
    y_tests = []
    clfs = []

    cnt = 0
    os.system("mkdir -p data/model/")

    for train, test in skf.split(X, y):

        X_trains.append(X.iloc[train])
        y_trains.append(y.iloc[train])
        X_tests.append(X.iloc[test])
        y_tests.append(y.iloc[test])

        clf = None

        if(classifier == "lr"):
            if(remove_column is None):
                store_file = "data/model/LR_" + dataset.name + "_" + \
                    str(dataset.config) + "_" + str(cnt) + \
                    "_" + str(fraction) + ".pkl"
            else:
                store_file = "data/model/LR_" + dataset.name + "_remove_" + remove_column.replace(
                    " ", "_") + "_" + str(dataset.config) + "_" + str(cnt) + "_" + str(fraction) + ".pkl"

            if(not os.path.isfile(store_file)):
                #  For linear classifier, we use Logistic regression model of sklearn
                clf = LogisticRegression(
                    class_weight='balanced', solver='liblinear', random_state=0)
                clf.fit(X_trains[-1], y_trains[-1])

                # save the classifier
                with open(store_file, 'wb') as fid:
                    pickle.dump(clf, fid)

            else:
                # Load the classifier
                with open(store_file, 'rb') as fid:
                    clf = pickle.load(fid)

        elif(classifier == "svm-linear"):
            if(remove_column is None):
                store_file = "data/model/SVM_" + dataset.name + "_" + \
                    str(dataset.config) + "_" + str(cnt) + \
                    "_" + str(fraction) + ".pkl"
            else:
                store_file = "data/model/SVM_" + dataset.name + "_remove_" + remove_column.replace(
                    " ", "_") + "_" + str(dataset.config) + "_" + str(cnt) + "_" + str(fraction) + ".pkl"
            if(not os.path.isfile(store_file)):
                #  For linear classifier, we use Logistic regression model of sklearn
                clf = SVC(kernel="linear")
                clf.fit(X_trains[-1], y_trains[-1])

                # save the classifier
                with open(store_file, 'wb') as fid:
                    pickle.dump(clf, fid)

            else:
                # Load the classifier
                with open(store_file, 'rb') as fid:
                    clf = pickle.load(fid)

        else:
            raise ValueError(classifier)

        clfs.append(clf)

        if(verbose):
            print("\nFeatures: ", X_trains[-1].columns.to_list())
            print("Number of features:", len(X_trains[-1].columns.to_list()))
            print("\nWeights: ", clf.coef_[0])
            print("\nBias:", clf.intercept_[0])
            assert len(clf.coef_[0]) == len(
                X_trains[-1].columns), "Error: wrong dimension of features and weights"

            print("Train Accuracy Score: ", clf.score(
                X_trains[-1], y_trains[-1]), "positive ratio: ", y_trains[-1].mean())
            print("Test Accuracy Score: ", clf.score(
                X_tests[-1], y_tests[-1]), "positive ratio: ", y_tests[-1].mean())

        cnt += 1

    if(compute_equalized_odds):
        return clfs, X_trains, X_tests, dataset.known_sensitive_attributes, y_trains, y_tests

    return clfs, X_trains, X_tests, dataset.known_sensitive_attributes
