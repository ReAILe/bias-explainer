from argparse import ArgumentError
from itertools import product
from fairxplainer import hdmr
from scipy.stats import norm, binom
import pandas as pd
import math
import numpy as np
import multiprocessing
# from copy import deepcopy
from fairxplainer.backfitting import KernelSmoothing
import random


class FairXplainer():
    def __init__(self, classifier, dataset, sensitive_features, label=None, verbose=False):
        self.classifier = classifier
        self.dataset = dataset
        self.sensitive_features = sensitive_features
        self.Y_ground_truth = label  # required for predictive parity
        self.random_bit_perturbation = False  # flag for an approximate result
        self.is_degenerate_case = False
        self.timeout = False
        if (verbose):
            print("Shape of feature matrix:", self.dataset.shape)
            if (label is not None):
                print("Shape of Y:", label.shape)

        for column in self.dataset.columns:
            if ("/" in column):
                self.dataset = self.dataset.rename(
                    {column: column.replace("/", " Or ")}, axis=1)  # "/" is used to show intersectional effects and hence reserved
                
    def compute(self, maxorder=2, lambax=0.01, spline_intervals=2, explain_sufficiency_fairness=False,
                approach="hdmr", # options: {hdmr, kernel}. hdmr is computationally efficient.
                compute_sp_only=False,
                verbose=False, seed=22, cpu_time=300):
        """
            This code computes the variance of individual and intersectional features, the sum of which 
            equals the overall variance of the prediction of the classifier. 
            Variance computation is grouped by each sensitive group
        """

        
        self.group_specific_positive_prediction_probabilities = {} # contains the probability of positive prediction of the classifier for a sensitive group
        self.group_specific_positive_prediction_probabilities_on_dataset = {} # contains the probability of positive prediction of the classifier for a sensitive group computed on the original dataset
        self.sensitive_groups = []
        self.group_specific_variance = {} # contains the total variance for a sensitive group
        sensitive_features_in_data = [] # contains the set of sensitive features in the dataset
        unique_sensitive_values = []
        idx_sensitive_feature_by_group = []

        """
            In our formulation, we consider multiple (categorical) sensitive features. Each feature can have more than two unique values. 
            Since categorical features with more than two values are one-hot encoded during training, we require to process them here. 
            In particular we assume a specific design pattern in this context. Let "Race" be a sensitive feature, which takes three values: Black, 
            White, and Colored. We assume the provided dataset has three columns, named as Race_Black, Race_White, and Race_Colored, each separated by underscore "_".

            The following code computes the unique values of each sensitive feature. Also, in case of more than two values for a sensitive feature, 
            we cluster indices of corresponding one-hot encoded features. 
        """
        i = 0
        for sensitive_feature in self.sensitive_features:
            if (sensitive_feature in self.dataset.columns):  # sensitive feature with binary values
                sensitive_features_in_data.append(sensitive_feature)
                idx_sensitive_feature_by_group.append([i])
                i += 1
                unique_sensitive_values.append(
                    list(self.dataset[sensitive_feature].unique()))
            else:
                idx_new_group = []
                for feature in self.dataset.columns:
                    # the case where feature belongs to a multi-valued (>2) sensitive feature.
                    if (feature.startswith(sensitive_feature)):
                        # print(feature)
                        sensitive_features_in_data.append(feature)
                        idx_new_group.append(i)
                        i += 1
                        unique_sensitive_values.append(
                            list(self.dataset[feature].unique()))
                assert len(idx_new_group) > 0
                idx_sensitive_feature_by_group.append(idx_new_group)

        # Overriding the definition of sensitive features based on the provided dataset
        self.sensitive_features = sensitive_features_in_data

        self.result = pd.DataFrame()
        self._auxiliary_info = {}
        all_features = list(self.dataset.columns)
        all_samples = None

        all_valid_assignments = []  # assignment of sensitive features
        assignment_samples = []
        assignment_Ys = []
        assignment_problems = []

        for idx, assignment in enumerate(list(product(*unique_sensitive_values))):

            # an invalid assignment is where at least two values of a sensitive feature is true for a categorical sensitive feature with more than 2 values
            invalid_assignment = False
            for cluster in idx_sensitive_feature_by_group:
                s = 0
                for i in cluster:
                    s += assignment[i]
                if (len(cluster) > 1 and s != 1):
                    invalid_assignment = True
                    continue
            if (invalid_assignment):
                continue
            
            # get statistics for the current assignment of sensitive features
            flag, dists, bounds, positive_prediction_probability, samples, Y_ground_truth = self._get_bounds_conditioned(
                assignment, apply_filtering=True)

            # rewriting positive prediction probability on a dataset
            if (explain_sufficiency_fairness):
                if (Y_ground_truth is None):
                    positive_prediction_probability = 0
                else:
                    positive_prediction_probability = Y_ground_truth.mean()

            if (not flag):
                continue

            # rewrite in pretty format
            self.sensitive_groups.append((", ").join(
                [str(a) + " = " + str(b) for a, b in zip(self.sensitive_features, assignment)]))
            self.group_specific_positive_prediction_probabilities_on_dataset[
                self.sensitive_groups[-1]] = positive_prediction_probability

            # SALib problem instance
            problem = {
                'num_vars': len(all_features),
                'names': all_features,
                'bounds': bounds,
                'dists': dists
            }

            if (assignment == "base"):
                assert all_samples is not None
                samples = all_samples
            else:
                if (all_samples is None):
                    all_samples = samples
                else:
                    all_samples = np.concatenate((all_samples, samples))

            # Model prediction
            if (approach in ["hdmr", "kernel"] and explain_sufficiency_fairness):
                assert self.Y_ground_truth is not None
                Y = Y_ground_truth
            else:
                Y = self.classifier.predict(samples)

            self.group_specific_positive_prediction_probabilities[self.sensitive_groups[-1]] = Y.mean(
            )
            self.group_specific_variance[self.sensitive_groups[-1]] = Y.var()

            if (assignment == "base"):
                raise ValueError()

            # store
            all_valid_assignments.append(assignment)
            assignment_samples.append(samples)
            assignment_problems.append(problem)
            assignment_Ys.append(Y)

        self._all_positive_prediction_probabilities = np.array(
            list(self.group_specific_positive_prediction_probabilities.values()))
        self._max_positive_prediction_probability_index = self._all_positive_prediction_probabilities.argmax()
        self._min_positive_prediction_probability_index = self._all_positive_prediction_probabilities.argmin()

        # Degenerate case
        if (self._all_positive_prediction_probabilities[self._max_positive_prediction_probability_index] ==
                self._all_positive_prediction_probabilities[self._min_positive_prediction_probability_index]):

            if (self._all_positive_prediction_probabilities[self._max_positive_prediction_probability_index] in [0, 1]):
                if (verbose):
                    print("each FIF is Zero")
                return

        if (compute_sp_only):  # only computes statistical parity
            return

        for idx, assignment in enumerate(all_valid_assignments):
            # enumerate for the most and least favorable sensitive groups

            if (self._max_positive_prediction_probability_index != self._min_positive_prediction_probability_index and idx not in [self._max_positive_prediction_probability_index, self._min_positive_prediction_probability_index]):
                if (verbose > 1):
                    print("\n\n\nc skipping ", self.sensitive_groups[idx])
                continue
            else:
                if (verbose):
                    print("\n\n\nc Decomposition for ",
                          self.sensitive_groups[idx])
                    print("c positive prediction probability of ",
                          self.sensitive_groups[idx], "is", self.group_specific_positive_prediction_probabilities[self.sensitive_groups[idx]])

            # restore
            samples = assignment_samples[idx]
            Y = assignment_Ys[idx]
            problem = assignment_problems[idx]
            assert assignment not in self.result

            if (approach in ["hdmr", "kernel"]):

                if (approach == "hdmr"):

                    # For hdmr approach, repeat samples to achieve min sample requirement of 300
                    if (samples.shape[0] < 300 and approach == "hdmr"):
                        """
                            This is not an ideal approach to repeat the dataset to meet min sample requirement
                        """
                        assert samples.shape[0] == Y.shape[0]
                        num_repeat = int(math.ceil(300.0 / samples.shape[0]))
                        samples = np.repeat(
                            samples, [num_repeat] * samples.shape[0], axis=0)
                        Y = np.repeat(Y, [num_repeat] * Y.shape[0], axis=0)

                    if (verbose):
                        print("c sample shape:", samples.shape)
                        print("c sensitive group:", self.sensitive_groups[idx])
                        print("c variance of Y", Y.var())
                        print("c mean of Y", Y.mean())

                    qu = multiprocessing.Queue()
                    pr = multiprocessing.get_context("fork").Process(target=hdmr.analyze, args=(
                        qu, problem, samples, Y, maxorder, 10000, spline_intervals, 1, None, 0.95, lambax, False, seed, ))
                    pr.daemon = True
                    pr.start()
                    pr.join(timeout=int(cpu_time/2))
                    pr.terminate()
                    while pr.exitcode == None:
                        pass

                    if pr.exitcode == 0:
                        pass
                        [result_current, rmse] = qu.get()
                    else:
                        self.timeout = True
                        raise RuntimeError("Timeout")

                else:
                    kernel_smoothing = KernelSmoothing(problem)
                    result_current = kernel_smoothing.analyze_step_by_step(
                        samples, Y)

                    if (verbose):
                        print(result_current)

                assert approach == "hdmr"
                # process result_current
                var_Y = Y.var()
                result_current.reset_index(inplace=True)
                result_current = result_current.rename(
                    {'index': 'names'}, axis=1)
                result_current['Var1'] = result_current.apply(
                    lambda x: (x['S']) if "/" not in x['names'] else np.NaN, axis=1)
                result_current['Var2'] = result_current.apply(
                    lambda x: (x['S']) if "/" in x['names'] else np.NaN, axis=1)
                result_current['VarTotal'] = result_current.apply(
                    lambda x: var_Y if "/" not in x['names'] else np.NaN, axis=1)
                result_current['structural contribution'] = result_current.apply(
                    lambda x: x['Sa'], axis=1)
                result_current['correlative contribution'] = result_current.apply(
                    lambda x: x['S'] - x['Sa'], axis=1)

                # print(result_current[['Sa', 'Sb', 'S']])

            else:
                raise NotImplementedError()

            # Store auxiliary information
            if ("decomposed_Y" not in self._auxiliary_info):
                self._auxiliary_info['decomposed_Y'] = {}
            if ("rmse" not in self._auxiliary_info):
                self._auxiliary_info['rmse'] = {}
            if ("Y" not in self._auxiliary_info):
                self._auxiliary_info['Y'] = {}
            if ("X" not in self._auxiliary_info):
                self._auxiliary_info['X'] = {}

            # Store results
            if (assignment == "base"):
                result_current['sensitive group'] = "base"
                if (approach == "hdmr"):
                    # self._auxiliary_info['decomposed_Y']['base'] = Y_estimated_decomposed
                    self._auxiliary_info['decomposed_Y']['base'] = None
                    self._auxiliary_info['rmse']['base'] = rmse
                    self._auxiliary_info['Y']['base'] = Y
                    self._auxiliary_info['X']['base'] = samples
            else:
                result_current['sensitive group'] = [
                    self.sensitive_groups[idx]]*result_current.shape[0]
                if (approach == "hdmr"):
                    # self._auxiliary_info['decomposed_Y'][self.sensitive_groups[idx]] = Y_estimated_decomposed
                    self._auxiliary_info['decomposed_Y'][self.sensitive_groups[idx]] = None
                    self._auxiliary_info['rmse'][self.sensitive_groups[idx]] = rmse
                    self._auxiliary_info['Y'][self.sensitive_groups[idx]] = Y
                    self._auxiliary_info['X'][self.sensitive_groups[idx]] = samples

            self.result = self.result.append(
                result_current, ignore_index=False)

        if (verbose):
            print()
            print(self.result)

    def get_weights(self):

        """
            Returns the weights of the features. This is called after compute() is called.
            Only shows first order influences. Second and higher order influences are summed up.
        """

        result = {}

        if (self.is_degenerate_case):
            return None

        if (self._all_positive_prediction_probabilities[self._max_positive_prediction_probability_index] ==
                self._all_positive_prediction_probabilities[self._min_positive_prediction_probability_index]):

            if (self._all_positive_prediction_probabilities[self._max_positive_prediction_probability_index] in [0, 1]):
                return pd.DataFrame(result.values(), columns=[
                    'weight'], index=result.keys())

        # First order
        first_order_effect_sum = 0
        for feature in self.result[self.result['VarTotal'].notnull()]['names'].unique():

            # get variance of majority group
            weight_majority_group = self.result[(self.result['sensitive group'] == self.sensitive_groups[self._max_positive_prediction_probability_index]) & (
                self.result['names'] == feature)]['Var1'].item()
            
            # scaling weight by the probability of the majority group
            weight_majority_group /= (
                1 - self.group_specific_positive_prediction_probabilities[self.sensitive_groups[self._max_positive_prediction_probability_index]])

            # get variance of minority group
            weight_minority_group = self.result[(self.result['sensitive group'] == self.sensitive_groups[self._min_positive_prediction_probability_index]) & (
                self.result['names'] == feature)]['Var1'].item()
            
            # scaling weight by the probability of the minority group
            weight_minority_group /= (
                1 - self.group_specific_positive_prediction_probabilities[self.sensitive_groups[self._min_positive_prediction_probability_index]])
          
            result[feature] = weight_majority_group - weight_minority_group

            first_order_effect_sum += result[feature]

        feature = None
        # second and higher order
        second_order_effect_sum = 0
        if ('Var2' in self.result.columns):
            for feature_subset in self.result[self.result['Var2'].notnull()]['names'].unique():

                # get variance of majority group
                weight_majority_group = self.result[(self.result['sensitive group'] == self.sensitive_groups[self._max_positive_prediction_probability_index]) & (
                    self.result['names'] == feature_subset)]['Var2'].item()
                
                # scaling weight by the probability of the majority group
                weight_majority_group /= (
                    1 - self.group_specific_positive_prediction_probabilities[self.sensitive_groups[self._max_positive_prediction_probability_index]])

                # get variance of minority group
                weight_minority_group = self.result[(self.result['sensitive group'] == self.sensitive_groups[self._min_positive_prediction_probability_index]) & (
                    self.result['names'] == feature_subset)]['Var2'].item()
                
                # scaling weight by the probability of the minority group
                weight_minority_group /= (
                    1 - self.group_specific_positive_prediction_probabilities[self.sensitive_groups[self._min_positive_prediction_probability_index]])

                second_order_effect_sum += weight_majority_group - weight_minority_group

        result = pd.DataFrame(result.values(), columns=[
                              'weight'], index=result.keys())
        result = result.sort_values('weight', ascending=False, key=abs)

        result.loc["FIFs " + r"$(\lambda > 1)$"] = second_order_effect_sum
        return result

    def get_top_k_weights(self, k=None):

        """
            Returns the top k weights of the features. This is called after compute() is called.
        """

        result = {}

        if (self.is_degenerate_case):
            return None

        if (self._all_positive_prediction_probabilities[self._max_positive_prediction_probability_index] ==
                self._all_positive_prediction_probabilities[self._min_positive_prediction_probability_index]):

            if (self._all_positive_prediction_probabilities[self._max_positive_prediction_probability_index] in [0, 1]):
                return pd.DataFrame(result.values(), columns=[
                    'weight'], index=result.keys())

        # First order
        first_order_effect_sum = 0
        for feature in self.result[self.result['VarTotal'].notnull()]['names'].unique():

            # get variance of majority group
            weight_majority_group = self.result[(self.result['sensitive group'] == self.sensitive_groups[self._max_positive_prediction_probability_index]) & (
                self.result['names'] == feature)]['Var1'].item()
            
            # scaling the weight by the probability of the majority group
            weight_majority_group /= (
                1 - self.group_specific_positive_prediction_probabilities[self.sensitive_groups[self._max_positive_prediction_probability_index]])

            # get variance of minority group
            weight_minority_group = self.result[(self.result['sensitive group'] == self.sensitive_groups[self._min_positive_prediction_probability_index]) & (
                self.result['names'] == feature)]['Var1'].item()
            
            # scaling the weight by the probability of the minority group
            weight_minority_group /= (
                1 - self.group_specific_positive_prediction_probabilities[self.sensitive_groups[self._min_positive_prediction_probability_index]])

            result[feature] = weight_majority_group - weight_minority_group

            first_order_effect_sum += result[feature]

        # second order
        second_order_effect_sum = 0
        if ('Var2' in self.result.columns):
            for feature_subset in self.result[self.result['Var2'].notnull()]['names'].unique():
                assert feature_subset not in result

                # get variance of majority group
                weight_majority_group = self.result[(self.result['sensitive group'] == self.sensitive_groups[self._max_positive_prediction_probability_index]) & (
                    self.result['names'] == feature_subset)]['Var2'].item()
                
                # scaling the weight by the probability of the majority group
                weight_majority_group /= (
                    1 - self.group_specific_positive_prediction_probabilities[self.sensitive_groups[self._max_positive_prediction_probability_index]])

                # get variance of minority group
                weight_minority_group = self.result[(self.result['sensitive group'] == self.sensitive_groups[self._min_positive_prediction_probability_index]) & (
                    self.result['names'] == feature_subset)]['Var2'].item()
                
                # scaling the weight by the probability of the minority group
                weight_minority_group /= (
                    1 - self.group_specific_positive_prediction_probabilities[self.sensitive_groups[self._min_positive_prediction_probability_index]])

                
                result[feature_subset] = weight_majority_group - weight_minority_group

                second_order_effect_sum += result[feature_subset]

        result = pd.DataFrame(result.values(), columns=[
                              'weight'], index=result.keys())
        result = result.sort_values('weight', ascending=False, key=abs)

        # get top k values
        if (k is not None):
            assert k >= 1
            total_weight = result.sum().item()
            result = result.head(k)
            top_k_weight = result.sum().item()
            residual_weight = total_weight - top_k_weight

        # result = result.sort_values('weight', ascending=False)

        if (k is not None):
            result.loc['Residual FIFs'] = residual_weight

        return result



    def _compute_statistical_parity(self, X, y, all_assignments):
        """
            Provided a feature matrix X and prediction Y, this code computes the probability of positive prediction
            of the classifier for all possible assignments of sensitive groups in X
        """
        assert "target" not in self.dataset.columns  # "target" is the name of class label in pandas
        assert len(all_assignments) > 0  # there is non-zero sensitive groups

        df = pd.DataFrame(np.column_stack((X, y)), columns=list(
            self.dataset.columns) + ['target'])

        """
            self.positive_prediction_probabilities is a dict where the key is a sensitive group and the value is a tuple. 
            In the tuple, the first value is the positive_prediction_probability and the second value is the sample weight of each sensitive group.
        """
        self.positive_prediction_probabilities = {}  # contains result
        for sensitive_features_assignment in all_assignments:
            mask = (True)
            for i in range(len(self.sensitive_features)):
                mask = mask & (df[self.sensitive_features[i]]
                               == sensitive_features_assignment[i])
            conditioned_df = df[mask]
            self.positive_prediction_probabilities[(", ").join([str(a) + " = " + str(b) for a, b in zip(self.sensitive_features, sensitive_features_assignment)])
                                                   ] = (conditioned_df['target'].mean(), conditioned_df.shape[0] / df.shape[0])

    

    def _get_bounds_conditioned(self, sensitive_features_assignment, apply_filtering=True, compute_sp=True, apply_kde=True):
        """
            Computes probability distribution, which is fed to SaLib library

            This code construct a conditioned dataset based on an assignment to the sensitive features.
            For each conditioned dataset, it computes the probability distribution of individual features.
        """
        conditioned_df = None
        conditioned_Y = None
        if (apply_filtering):
            mask = (True)
            for i in range(len(self.sensitive_features)):
                mask = mask & (
                    self.dataset[self.sensitive_features[i]] == sensitive_features_assignment[i])
            conditioned_df = self.dataset[mask]
            if (self.Y_ground_truth is not None):
                conditioned_Y = self.Y_ground_truth[mask].values
        else:
            conditioned_df = self.dataset
            if (self.Y_ground_truth is not None):
                conditioned_Y = self.Y_ground_truth.values

        dists = []  # name of the distribution
        bounds = []  # parameter of the distribution
        for column in conditioned_df.columns:
            if (len(conditioned_df[column].unique()) <= 2):
                dists.append("bernoulli")
                p = conditioned_df[column].mean()
                bounds.append([p, None])
            else:
                # apply_kde = False
                if (apply_kde):
                    # univariate kde
                    dists.append("kde")
                    bounds.append([conditioned_df[column].values, None])
                else:
                    dists.append("norm")
                    mu, std = norm.fit(conditioned_df[column].values)
                    bounds.append([mu, std])

        assert len(dists) == len(bounds)
        if (conditioned_df.shape[0] > 0):

            # computes the probability of positive prediction of the conditioned dataset
            if (compute_sp):
                return True, dists, bounds, self.classifier.predict(conditioned_df.values).mean(), conditioned_df.values, conditioned_Y

            return True, dists, bounds, None, conditioned_df.values, conditioned_Y
        else:
            if (compute_sp):
                return False, None, None, None, None, None

            return False, None, None, None, None, None

    
    def statistical_parity_dataset(self, verbose=False):
        self._all_positive_prediction_probabilities_on_dataset = np.array(
            list(self.group_specific_positive_prediction_probabilities_on_dataset.values()))
        if (verbose):
            print("Exact statistical parity:", self._all_positive_prediction_probabilities_on_dataset.max(
            ) - self._all_positive_prediction_probabilities_on_dataset.min())
            print("="*50)
        return self._all_positive_prediction_probabilities_on_dataset.max() - self._all_positive_prediction_probabilities_on_dataset.min()

    
    def statistical_parity_sample(self, verbose=False):
        self._all_positive_prediction_probabilities = np.array(
            list(self.group_specific_positive_prediction_probabilities.values()))
        if (verbose):
            print("Empirical statistical parity:", self._all_positive_prediction_probabilities.max(
            ) - self._all_positive_prediction_probabilities.min())
            print("="*50)
        return self._all_positive_prediction_probabilities.max() - self._all_positive_prediction_probabilities.min()


"""
============================================================
For Plotting results in a waterfall diagram
"""


def plot(result, draw_waterfall=True, fontsize=22, labelsize=18, figure_size=(10, 7), title="", xlim=None,
         x_label="Statistical Parity", text_x_pad=0.02, text_y_pad=0.1, result_x_pad=0.02, result_y_location=0.5, delete_zero_weights=False):

    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = figure_size

    assert "weight" in result.columns
    assert len(result.columns) == 1

    if (delete_zero_weights):
        # Delete 0 weight rows
        result = result.drop(result[result['weight'] == 0].index)

    # rename features
    result['feature'] = result.index
    result['feature'] = result.apply(lambda x: x['feature'].replace("/", " \& ").replace("_", " ") if isinstance(x['feature'], str)
                                     else (" \& ".join(x['feature']).replace("_", " ") if isinstance(x['feature'], tuple) else None), axis=1)
    result.index = result['feature']
    del result['feature']
    colormap = result.apply(
        lambda x: 'red' if x['weight'] > 0 else "green", axis=1)

    if (draw_waterfall):

        blank = result['weight'].cumsum().shift(1).fillna(0)
        step = blank.reset_index(drop=True).repeat(3).shift(-1)
        step[1::3] = np.nan

        # Bar
        my_plot = result['weight'].plot(
            kind='barh', stacked=True, left=blank, legend=None, color=colormap)
        my_plot.plot(step.values, step.index, 'black')

        # Bar value
        for i in range(result.shape[0]):
            if (result['weight'].iloc[i] == 0):
                my_plot.text(result['weight'].iloc[i] + blank.iloc[i] + (plt.xlim()[1] - plt.xlim()[0]) * text_x_pad,
                             ((plt.ylim()[1] - plt.ylim()[0]) / result.shape[0]) * i + text_y_pad, str(0), color='green', fontsize=labelsize)
            elif (result['weight'].iloc[i] > 0):
                my_plot.text(result['weight'].iloc[i] + blank.iloc[i] + (plt.xlim()[1] - plt.xlim()[0]) * text_x_pad,
                             ((plt.ylim()[1] - plt.ylim()[0]) / result.shape[0]) * i + text_y_pad, str(round(result['weight'].iloc[i], 3)), color='red', fontsize=labelsize)
            else:
                my_plot.text(blank.iloc[i] + (plt.xlim()[1] - plt.xlim()[0]) * text_x_pad,
                             ((plt.ylim()[1] - plt.ylim()[0]) / result.shape[0]) * i + text_y_pad, str(round(result['weight'].iloc[i], 3)), color='green', fontsize=labelsize)

        # fairness value
        fairness_value = result.sum().item()
        my_plot.axvline(x=fairness_value, color='black', linestyle='--')
        my_plot.text(fairness_value + (plt.xlim()[1] - plt.xlim()[0]) * result_x_pad, (plt.ylim()[1] - plt.ylim()[0])
                     * result_y_location, str(round(fairness_value, 3)), color='blue', rotation=0, fontsize=labelsize)

    else:

        # Bar
        my_plot = result['weight'].plot(
            kind='barh', legend=None, color=colormap)
        # my_plot.axvline(x=0, color='gray')
        plt.grid(axis='x')

    my_plot.invert_yaxis()
    mn, mx = plt.xlim()
    plt.xlim(mn - 0.1 * (mx - mn), mx + 0.1 * (mx - mn))
    plt.xticks(fontsize=labelsize, rotation=0)
    plt.yticks(fontsize=labelsize)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel("", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    if (xlim is not None and isinstance(xlim, tuple)):
        plt.xlim(xlim[0], xlim[1])
    plt.tight_layout()

    return plt
