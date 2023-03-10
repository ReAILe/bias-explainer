# this script learns a decision tree classifier,
# generate a DNF rule for the positive class (in the standard binary classification setting),
# negate the DNF to CNF


from sklearn import tree
from sklearn.tree import _tree
from sklearn.model_selection import KFold
from sklearn import metrics
from fairxplainer import utils
import os
import pickle





def init(dataset, repaired=False, verbose=False, compute_equalized_odds=False, depth=5, remove_column=None):

    df = dataset.get_df(repaired=repaired)

    # get X,y
    X = df.drop(['target'], axis=1)
    y = df['target']

    if(remove_column is not None):
        assert isinstance(remove_column, str)
        X = X.drop([remove_column], axis=1)

    # one-hot
    X = utils.get_one_hot_encoded_df(
        X, dataset.categorical_attributes, verbose=verbose)

    skf = KFold(n_splits=5, shuffle=True, random_state=10)
    skf.get_n_splits(X, y)

    X_trains = []
    y_trains = []
    X_tests = []
    y_tests = []
    clfs = []

    os.system("mkdir -p data/model/")
    cnt = 0
    for train, test in skf.split(X, y):

        X_trains.append(X.iloc[train])
        y_trains.append(y.iloc[train])
        X_tests.append(X.iloc[test])
        y_tests.append(y.iloc[test])

        if(remove_column is None):
            store_file = "data/model/DT_" + dataset.name + "_" + \
                str(dataset.config) + "_" + \
                str(depth) + "_" + str(cnt) + ".pkl"
        else:
            store_file = "data/model/DT_" + dataset.name + "_remove_" + remove_column.replace(
                " ", "_") + "_" + str(dataset.config) + "_" + str(depth) + "_" + str(cnt) + ".pkl"

        if(not os.path.isfile(store_file)):
            if(depth < 0):
                clf = tree.DecisionTreeClassifier(max_depth=None)
            else:
                clf = tree.DecisionTreeClassifier(max_depth=depth)
            clf.fit(X_trains[-1], y_trains[-1])
            tree_preds = clf.predict_proba(X_tests[-1])[:, 1]

            # save the classifier
            with open(store_file, 'wb') as fid:
                pickle.dump(clf, fid)

        else:
            # Load the classifier
            with open(store_file, 'rb') as fid:
                clf = pickle.load(fid)

        clfs.append(clf)

        # clf = tree.DecisionTreeClassifier()
        # clf = clf.fit(X_train, y_train)
        predict_train = clf.predict(X_trains[-1])
        predict_test = clf.predict(X_tests[-1])

        if(verbose):
            print("\nTrain accuracy:", metrics.accuracy_score(
                y_trains[-1], predict_train), "positive ratio: ", y_trains[-1].mean())
            print("Test accuracy:", metrics.accuracy_score(
                y_tests[-1], predict_test), "positive ratio: ", y_tests[-1].mean())
            print("Train set positive prediction", predict_train.mean())
            print("Test set positive prediction", predict_test.mean())

        cnt += 1

    if(compute_equalized_odds):
        return clfs, X_trains, X_tests, dataset.known_sensitive_attributes, y_trains, y_tests

    return clfs, X_trains, X_tests, dataset.known_sensitive_attributes
