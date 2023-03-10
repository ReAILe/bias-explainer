from tokenize import group
from unittest import result
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from shap.plots import colors
from itertools import product
import multiprocessing



class ShapExplanation():
    def __init__(self):
        pass


    def compute(self, clf, X, max_group_mask, min_group_mask, model_name, cpu_time=300, verbose=False):
        """
            Sensitive feature is strictly Boolean for this experiment
        """

        # implement timeout
        qu = multiprocessing.Queue()
        pr = multiprocessing.get_context("fork").Process(target=self._compute, args=(
            qu, clf, X, max_group_mask, min_group_mask, model_name, verbose,))
        pr.daemon = True
        pr.start()
        pr.join(timeout=int(cpu_time))
        pr.terminate()
        while pr.exitcode == None:
            pass

        if pr.exitcode == 0:
            pass
            [result, exact_sp] = qu.get()
            return result, exact_sp
        else:
            self.timeout = True
            raise RuntimeError("Timeout")

    
    def _compute(self, queue, clf, X, max_group_mask, min_group_mask, model_name, verbose=False):
        # print(model_name)
        preds = clf.predict(X)
        shap_values = None
        if(model_name == "nn"):
            # explainer = shap.Explainer(clf, X)
            explainer = shap.KernelExplainer(clf.predict, shap.sample(X, 10000))
            shap_values = explainer.shap_values(shap.sample(X, 100))
            # shap_values = explainer.shap_values(X)
        else:
            explainer = shap.Explainer(clf, X)
            shap_values = explainer.shap_values(X)
        if(isinstance(shap_values, list)):
            shap_values = shap_values[1]

        glabel = "Demographic parity difference\nof model output for women vs. men"
        xmin = -1.5
        xmax = 1.5
        result = self._group_difference(
            shap_values, max_group_mask, min_group_mask, X.columns, xmin=xmin, xmax=xmax, xlabel=glabel, verbose=verbose)
        
        queue.put([result, preds[max_group_mask].mean() - preds[min_group_mask].mean()])


    def _group_difference(self, shap_values, max_group_mask, min_group_mask, feature_names=None, xlabel=None, xmin=None, xmax=None,
                          max_display=None, sort=True, show=True, verbose=False):
        """ This plots the difference in mean SHAP values between two groups.

        It is useful to decompose many group level metrics about the model output among the
        input features. Quantitative fairness metrics for machine learning models are
        a common example of such group level metrics.

        Parameters
        ----------
        shap_values : numpy.array
            Matrix of SHAP values (# samples x # features) or a vector of model outputs (# samples).

        group_mask : numpy.array
            A boolean mask where True represents the first group of samples and False the second.

        feature_names : list
            A list of feature names.
        """

        # constrain to feature-specific explanation
        assert len(shap_values.shape) != 1

        # compute confidence bounds for the group difference value
        vs = []
        gmean = max_group_mask.mean()
        for i in range(200):
            r = np.random.rand(shap_values.shape[0]) > gmean
            vs.append(shap_values[r].mean(0) - shap_values[~r].mean(0))
        vs = np.array(vs)
        xerr = np.vstack([np.percentile(vs, 95, axis=0),
                          np.percentile(vs, 5, axis=0)])

        # See if we were passed a single model output vector and not a matrix of SHAP values
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1).T
            if feature_names == None:
                feature_names = [""]

        # fill in any missing feature names
        if feature_names is None:
            feature_names = ["Feature %d" %
                             i for i in range(shap_values.shape[1])]

        if(verbose):
            print(shap_values.mean(0).sum())
            print(shap_values[max_group_mask].mean(0).sum()*(max_group_mask.mean()))
            print(shap_values[min_group_mask].mean(
                0).sum()*(min_group_mask.mean()))
            print(max_group_mask.mean())
            print(shap_values[max_group_mask].mean(0).sum()*(max_group_mask.mean()) +
                  shap_values[min_group_mask].mean(0).sum()*(min_group_mask.mean()))

        # Assuming statistical parity as a positive quantity between 0 and 1
        if((shap_values[max_group_mask].mean(0) - shap_values[min_group_mask].mean(0)).sum() >= 0):
            diff = shap_values[max_group_mask].mean(
                0) - shap_values[min_group_mask].mean(0)
        else:
            diff = shap_values[min_group_mask].mean(
                0) - shap_values[max_group_mask].mean(0)

        if sort == True:
            inds = np.argsort(-np.abs(diff)).astype(int)
        else:
            inds = np.arange(len(diff))

        if max_display != None:
            inds = inds[:max_display]

        result = pd.DataFrame(
            {"weight": diff[inds]}, index=feature_names[inds])

        return result
        # draw the figure
        figsize = [6.4, 0.2 + 0.9 * len(inds)]
        pl.figure(figsize=figsize)
        ticks = range(len(inds)-1, -1, -1)
        pl.axvline(0, color="#999999", linewidth=0.5)
        pl.barh(
            ticks, diff[inds], color=colors.blue_rgb,
            capsize=3, xerr=np.abs(xerr[:, inds])
        )

        for i in range(len(inds)):
            pl.axhline(y=i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

        ax = pl.gca()
        ax.set_yticklabels([feature_names[i] for i in inds])
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('none')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(labelsize=11)
        if xlabel is None:
            xlabel = "Group SHAP value difference"
        ax.set_xlabel(xlabel, fontsize=13)
        pl.yticks(ticks, fontsize=13)
        xlim = list(pl.xlim())
        if xmin is not None:
            xlim[0] = xmin
        if xmax is not None:
            xlim[1] = xmax
        pl.xlim(*xlim)
        if show:
            pl.show()



# from FairXplainer
def _get_bounds_conditioned(classifier, dataset, sensitive_features, sensitive_features_assignment, apply_filtering=True):
    """
        This code construct a conditioned dataset based on an assignment to the sensitive features.
        For each conditioned dataset, it computes the distribution of individual features.
    """
    mask = None
    conditioned_df = None
    if(apply_filtering):
        mask = (True)
        for i in range(len(sensitive_features)):
            mask = mask & (
                dataset[sensitive_features[i]] == sensitive_features_assignment[i])
        conditioned_df = dataset[mask]
    else:
        conditioned_df = dataset

    if(conditioned_df.shape[0] > 0):

        # computes the PPV of the conditioned dataset
        return True, classifier.predict(conditioned_df.values).mean(), mask
    else:
        return False, None, None

def row_masking_based_on_sensitive_groups(classifier, dataset, sensitive_features):
    # contains the PPV for a sensitive group computed on the original dataset
    group_specific_positive_prediction_probabilities_on_dataset = {}
    group_mask = {}
    sensitive_groups = []

    sensitive_features_in_data = []
    unique_sensitive_values = []
    idx_sensitive_feature_by_group = []

    i = 0
    for sensitive_feature in sensitive_features:
        if(sensitive_feature in dataset.columns):  # the sensitive feature has binary values
            sensitive_features_in_data.append(sensitive_feature)
            idx_sensitive_feature_by_group.append([i])
            i += 1
            unique_sensitive_values.append(
                list(dataset[sensitive_feature].unique()))
        else:
            idx_new_group = []
            for feature in dataset.columns:
                # the case where feature belongs to a multi-valued (>2) sensitive feature.
                if(feature.startswith(sensitive_feature)):
                    # print(feature)
                    sensitive_features_in_data.append(feature)
                    idx_new_group.append(i)
                    i += 1
                    unique_sensitive_values.append(
                        list(dataset[feature].unique()))
            assert len(idx_new_group) > 0
            idx_sensitive_feature_by_group.append(idx_new_group)

    # Overriding the definition of sensitive features based on the provided dataset
    sensitive_features = sensitive_features_in_data

    for idx, assignment in enumerate(list(product(*unique_sensitive_values))):

        # an invalid assignment is where at least two values of a sensitive feature is true for a multi-valued sensitive feature
        invalid_assignment = False
        for cluster in idx_sensitive_feature_by_group:
            s = 0
            for i in cluster:
                s += assignment[i]
            if(len(cluster) > 1 and s != 1):
                invalid_assignment = True
                continue
        if(invalid_assignment):
            continue

        flag, positive_prediction_probability, mask = _get_bounds_conditioned(
            classifier, dataset, sensitive_features, assignment, apply_filtering=True)

        if(not flag):
            continue

        sensitive_groups.append((", ").join(
            [str(a) + " = " + str(b) for a, b in zip(sensitive_features, assignment)]))
        group_specific_positive_prediction_probabilities_on_dataset[
            sensitive_groups[-1]] = positive_prediction_probability
        group_mask[sensitive_groups[-1]] = mask.values

    # getting max-min
    all_positive_prediction_probabilities = np.array(
        list(group_specific_positive_prediction_probabilities_on_dataset.values()))
    max_positive_prediction_probability_index = all_positive_prediction_probabilities.argmax()
    min_positive_prediction_probability_index = all_positive_prediction_probabilities.argmin()


    # print(group_specific_positive_prediction_probabilities_on_dataset.values())
    # print(group_mask)
    # print(max_positive_prediction_probability_index, min_positive_prediction_probability_index)
    return group_mask[sensitive_groups[max_positive_prediction_probability_index]], group_mask[sensitive_groups[min_positive_prediction_probability_index]]