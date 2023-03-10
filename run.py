from fairxplainer.wrapper import decision_tree_wrap, linear_classifier_wrap, mlp_wrap
from data.objects.communities import Communities_and_Crimes
from data.objects.bank import Bank
from data.objects.compas import Compas
from data.objects.german import German
from data.objects.adult import Adult
from data.objects.titanic import Titanic
from data.objects.ricci import Ricci
from data.objects.law_school import Law_School
import os
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np
import argparse
from time import time
from fairxplainer.fair_explainer import FairXplainer
from fairxplainer.backfitting import KernelSmoothing
from fairxplainer.fair_explainer import plot as fif_plot
from fairxplainer.wrapper.shap_fairness_explanation import ShapExplanation, row_masking_based_on_sensitive_groups



parser = argparse.ArgumentParser()
parser.add_argument("--thread", help="index of thread", default=-1, type=int)
parser.add_argument("--max_thread", help="num of threads",
                    default=24, type=int)
parser.add_argument("--dataset", nargs='+', default=["compas"])
parser.add_argument("--model", default=["dt"], nargs='+')
parser.add_argument("--single_fold", action='store_true')
parser.add_argument("--train", action='store_true')
parser.add_argument("--config", type=int, default=1)
parser.add_argument("--spline_intervals", type=int, default=2)
parser.add_argument("--token", nargs='+', default=['train'])
parser.add_argument("--explainer", nargs='+', default=['fairXplainer'])
parser.add_argument("--fairness_metrics", nargs='+', default=['sp'])
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument("--fold", nargs='+', type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("--dt_depth", nargs='+', type=int, default=[5])
parser.add_argument("--fraction_features", type=float, default=1)
parser.add_argument("--exp_intervention", action='store_true')
parser.add_argument("--exp_sampling", action='store_true')
parser.add_argument("--maxorder", type=int, default=2)
args = parser.parse_args()


if(args.single_fold):
    args.fold = [0]

args.compute_independence_metric = True

combinations = []

configs = {
    'titanic': 6,
    'ricci': 2,
    'adult': 6,
    'compas': 6,
    'german': 2,
    # 'law_school': 2,
    # 'bank': 2,
    # 'communities' : 126,
}
for dataset in args.dataset:
    for _ in range(127):
        if(configs[dataset] > -1):
            combinations.append((dataset, configs[dataset]))
            configs[dataset] -= 1


if(args.thread == -1):
    combinations = [(dataset, args.config) for dataset in args.dataset]
# print(combinations)
# quit()


# multiply combinations for each explainer and model
temp_combinations = []
for combination in combinations:
    for model in args.model:
        for explainer in args.explainer:
            if(model != "dt"):
                temp_combinations.append(
                    (explainer, model, None, combination[0], combination[1]))
            else:
                for depth in args.dt_depth:
                    temp_combinations.append(
                        (explainer, model, depth, combination[0], combination[1]))


combinations = temp_combinations

verbose = args.verbose
# print(args.max_thread, args.thread)
for idx, combination in enumerate(combinations):
    if(args.thread == idx % args.max_thread or args.thread == -1):
        explainer, model_name, depth, args.dataset, args.config = combination
        print(combination)
        # continue

        datasetObj = None
        if(args.dataset == "titanic"):
            datasetObj = Titanic(verbose=verbose, config=args.config)
        elif(args.dataset == "compas"):
            datasetObj = Compas(verbose=verbose, config=args.config)
        elif(args.dataset == "ricci"):
            datasetObj = Ricci(verbose=verbose, config=args.config)
        elif(args.dataset == "adult"):
            datasetObj = Adult(verbose=verbose, config=args.config)
        elif(args.dataset == "law_school"):
            datasetObj = Law_School(verbose=verbose, config=args.config)
        elif(args.dataset == "german"):
            datasetObj = German(verbose=verbose, config=args.config)
        elif(args.dataset == "bank"):
            datasetObj = Bank(verbose=verbose, config=args.config)
        elif(args.dataset == "communities"):
            datasetObj = Communities_and_Crimes(
                verbose=verbose, config=args.config)

        else:
            raise ValueError(args.dataset + " is not a defined dataset")

        start_time_preprocessing = time()

        if(model_name == 'lr'):
            model, data_train, data_test, sensitive_attributes, y_train, y_test = linear_classifier_wrap.init(
                datasetObj, classifier=model_name, repaired=False, verbose=False, compute_equalized_odds=True, fraction=args.fraction_features)

        if(model_name == 'nn'):
            model, data_train, data_test, sensitive_attributes, y_train, y_test = mlp_wrap.init(
                datasetObj, classifier=model_name, repaired=False, verbose=False, compute_equalized_odds=True, fraction=args.fraction_features)

        if(model_name == 'svm-linear'):
            model, data_train, data_test, sensitive_attributes, y_train, y_test = linear_classifier_wrap.init(
                datasetObj, classifier=model_name, repaired=False, verbose=False, compute_equalized_odds=True, fraction=args.fraction_features)

        if(model_name == 'dt'):
            model, data_train, data_test, sensitive_attributes, y_train, y_test = decision_tree_wrap.init(
                datasetObj, repaired=False, verbose=False, compute_equalized_odds=True, depth=depth)

        # if(model_name == "CNF"):
        #     model, data_train, data_test, sensitive_attributes, y_train, y_test = mlic_wrap.init(
        #         datasetObj, repaired=False, verbose=False, compute_equalized_odds=True, thread=args.thread)

        end_time_preprocessing = time()

        if(args.train):
            continue

        # for i in tqdm(range(5)):
        for i in args.fold:
            if(args.verbose):
                print("CV index", i)
            for fairness_metric in args.fairness_metrics:

                for token in args.token:

                    if(token == "train"):
                        data = data_train
                        data_y = y_train
                    elif(token == "test"):
                        data = data_test
                        data_y = y_test
                    else:
                        raise ValueError
                    
                    # print(token)
                    # print(args.dataset)
                    # print(data[i].shape)
                    # quit()
                    # continue

                    saved_DI = None

                    result = pd.DataFrame()

                    # all results
                    entry = {}
                    for indv_entry in ['disentangled influence function (suff) (y=1)', 'estimated suff (y=0)', 'exact suff (y=0)',
                                       'token', 'estimated suff (y=1)', 'sensitive attributes (as list)', 'disentangled influence function (eo) (y=0)',
                                       'influence function (eo) (y=1)', 'disentangled influence function (suff) (y=0)', 'processing time',
                                       'exact statistical parity', 'exact eo (y=1)', 'influence function (eo) (y=0)', 'estimated eo (y=0)',
                                       'rmse', 'disentangled influence function (sp)', 'sensitive attributes', 'estimated eo (y=1)', 'exact suff (y=1)',
                                       'fairness metric', 'estimated statistical parity', 'cv index', 'model', 'dt_depth', 'influence function (sp)',
                                       'disentangled influence function (eo) (y=1)', 'dataset', 'time', 'exact eo (y=0)', 'thread', 'maxorder', 'spline intervals',
                                       'fairXplainer degenerate case', 'influence function (suff) (y=0)', 'explainer', 'influence function (suff) (y=1)', 'config', 'suboptimal solution (sp)']:
                        entry[indv_entry] = None
                    if(args.exp_intervention and model_name in ['lr']):    
                        entry['fifs'] = None
                        entry['modified sps'] = None
                        entry['intervention correlation'] = None
                        entry['intervention p-value'] = None
                        entry['cosine similarity'] = None
                                        
                    entry['fairXplainer degenerate case'] = False
                    entry['suboptimal solution (sp)'] = False
                    entry['config'] = args.config
                    entry['spline intervals'] = args.spline_intervals
                    entry['thread'] = 0
                    entry['dataset'] = args.dataset
                    entry['model'] = model_name
                    entry['processing time'] = end_time_preprocessing - \
                        start_time_preprocessing
                    entry['sensitive attributes'] = (
                        ",").join(sensitive_attributes)
                    entry['sensitive attributes (as list)'] = sensitive_attributes
                    entry['fairness metric'] = fairness_metric

                    if(verbose):
                        print("\n", explainer, "\n")

                    start_time = time()
                    if(explainer == "fairXplainer"):
                        try:
                            # Explanation of statistical parity
                            if(fairness_metric == "sp"):
                                if(not args.exp_sampling):
                                    fairXplainer = FairXplainer(
                                        model[i], data[i], sensitive_attributes, verbose=verbose)
                                    fairXplainer.compute(
                                        approach="hdmr", spline_intervals=args.spline_intervals, maxorder=args.maxorder, verbose=verbose)
                                    entry['exact statistical parity'] = fairXplainer.statistical_parity_dataset(
                                        verbose=verbose)
                                    entry['maxorder'] = args.maxorder
                                    if(not fairXplainer.is_degenerate_case):
                                        explanation_result = fairXplainer.get_top_k_weights(
                                            k=7)
                                        try:
                                            entry['rmse'] = list(fairXplainer._auxiliary_info['rmse'].values())
                                        except:
                                            entry['rmse'] = None
                                        entry['estimated statistical parity'] = explanation_result.sum(
                                        ).item()
                                        entry['influence function (sp)'] = explanation_result.to_dict(
                                            'index')
                                        entry['suboptimal solution (sp)'] = fairXplainer.random_bit_perturbation

                                        if(verbose):
                                            print(explanation_result)
                                            print()
                                            print("Statistical parity:",
                                                explanation_result.sum().item())
                                            print("Exact statistical parity:",
                                                entry['exact statistical parity'])

                                        # explanation_result = fairXplainer.get_top_k_weights_disentangled(
                                        #     k=10, verbose=verbose)
                                        # entry['disentangled influence function (sp)'] = explanation_result.to_dict(
                                        #     'index')


                                        if(args.exp_intervention and model_name in ['lr']):
                                            import copy
                                            fifs = []
                                            feature_subsets = []
                                            modified_sps = []
                                            ordered_feature_list = []
                                            explanation_result = fairXplainer.get_top_k_weights()
                                            for feature_subset in explanation_result.index[:10]:
                                                if(feature_subset in ["Residual FIFs", "FIFs $(\lambda > 1)$"]):
                                                    continue
                                                ordered_feature_list = feature_subset.split("/")
                                                ordered_feature_list = list(set(ordered_feature_list))
                                                
                                                clf_modified = copy.deepcopy(model[i])
                                                for feature in ordered_feature_list:
                                                    clf_modified.coef_[0][list(data[i].columns).index(feature)] = 0
                                                fairXplainer_modified = FairXplainer(clf_modified, data[i], sensitive_attributes)    

                                                fairXplainer_modified.compute(approach = 'hdmr', compute_sp_only=True, verbose=False, spline_intervals=4)
                                                # print(ordered_feature_list, fairXplainer_modified.statistical_parity_sample())
                                                modified_sps.append(fairXplainer_modified.statistical_parity_sample())
                                                feature_subsets.append((" \& ".join(ordered_feature_list)).replace("_", " "))
                                                fifs.append(explanation_result.loc[feature_subset]['weight'].item())
                                                if(len(ordered_feature_list) > 1):
                                                    for feature in ordered_feature_list:
                                                        # print(feature, fifs[-1], explanation_result.loc[feature]['weight'].item())
                                                        fifs[-1] += explanation_result.loc[feature]['weight'].item()
                                                
                                            
                                            entry['fifs'] = list(fifs)
                                            entry['modified sps'] = list(entry['exact statistical parity'] - modified_sps)
                                            entry['intervention correlation'], entry['intervention p-value'] = pearsonr(fifs, entry['exact statistical parity'] - modified_sps)
                                            entry['cosine similarity'] = np.dot(fifs, entry['exact statistical parity'] - modified_sps) / (np.linalg.norm(fifs) * np.linalg.norm(entry['exact statistical parity'] - modified_sps))

                                    else:
                                        entry['fairXplainer degenerate case'] = True
                                    
                                        

    


                                elif(args.exp_sampling):
                                    # in sampling experiment, we evaluate the robustness of fairness influence functions

                                    # key: sample_size, value: array of results where each element for one iteration for a fixed sample size
                                    exact_statistical_parity = {}
                                    estimated_statistical_parity = {}
                                    influence_function = {}
                                    suboptimal_solution = {}
                                    disentangled_influence = {}

                                    for sample_size in np.linspace(0.1, 1, 20):
                                        # for sample_size in [0.8]:

                                        sample_size = (sample_size, int(
                                            sample_size * data[i].shape[0]))
                                        exact_statistical_parity[sample_size] = []
                                        estimated_statistical_parity[sample_size] = [
                                        ]
                                        influence_function[sample_size] = []
                                        suboptimal_solution[sample_size] = []
                                        disentangled_influence[sample_size] = []

                                        for sampling_iteration in range(10):
                                            # print(sample_size)
                                            # print(data[i].shape)
                                            np.random.seed()
                                            sampled_data = data[i].sample(
                                                n=sample_size[1], random_state=None)

                                            # print(sampled_data)
                                            # print()
                                            # print(sampled_data.describe())
                                            # print()

                                            fairXplainer = FairXplainer(
                                                model[i], sampled_data, sensitive_attributes, verbose=verbose)
                                            fairXplainer.compute(
                                                approach="hdmr", spline_intervals=args.spline_intervals, verbose=verbose)
                                            explanation_result = fairXplainer.get_top_k_weights(
                                                k=10)

                                            exact_statistical_parity[sample_size].append(
                                                fairXplainer.statistical_parity_dataset(verbose=verbose))
                                            estimated_statistical_parity[sample_size].append(
                                                explanation_result.sum().item())
                                            influence_function[sample_size].append(
                                                explanation_result.to_dict('index'))
                                            suboptimal_solution[sample_size].append(
                                                fairXplainer.random_bit_perturbation)

                                            if(verbose):
                                                print(explanation_result)
                                                print("estimated statistical parity:",
                                                    explanation_result.sum().item())

                                            # explanation_result = fairXplainer.get_top_k_weights_disentangled(
                                            #     k=10, verbose=verbose)

                                            # disentangled_influence[sample_size].append(
                                            #     explanation_result.to_dict('index'))

                                            # print(explanation_result)

                                    entry['exact statistical parity'] = exact_statistical_parity
                                    entry['estimated statistical parity'] = estimated_statistical_parity
                                    entry['influence function (sp)'] = influence_function
                                    entry['suboptimal solution (sp)'] = suboptimal_solution
                                    entry['disentangled influence function (sp)'] = disentangled_influence

                                    if(verbose):
                                        print()
                                        print("Statistical parity:",
                                            entry['estimated statistical parity'])
                                        print("Exact statistical parity:",
                                            entry['exact statistical parity'])

                            # Explanation of equalized odds
                            elif(fairness_metric == "eo"):
                                # print(data[i].shape)
                                y_unique = data_y[i].unique()
                                if(not (len(y_unique) == 2 and 1 in y_unique and 0 in y_unique)):
                                    # raise ValueError()
                                    continue
                                for y_val in [0, 1]:
                                    if(verbose):
                                        print("\n", "%"*50)
                                        print("Y = ", y_val)

                                    if(y_val == 1):
                                        fairXplainer = FairXplainer(
                                            model[i], data[i][data_y[i] == 1], sensitive_attributes, verbose=verbose)
                                    else:
                                        fairXplainer = FairXplainer(
                                            model[i], data[i][data_y[i] == 0], sensitive_attributes, verbose=verbose)

                                    fairXplainer.compute(
                                        approach="hdmr", spline_intervals=args.spline_intervals, maxorder=args.maxorder, verbose=verbose)
                                    entry['maxorder'] = args.maxorder
                                    entry['exact eo (y=' + str(y_val) + ")"] = fairXplainer.statistical_parity_dataset(
                                        verbose=verbose)

                                    if(not entry['fairXplainer degenerate case'] and not fairXplainer.is_degenerate_case):
                                        explanation_result = fairXplainer.get_top_k_weights(
                                            k=10)
                                        entry['estimated eo (y=' + str(y_val) +
                                            ")"] = explanation_result.sum().item()
                                        entry['influence function (eo) (y=' + str(
                                            y_val) + ")"] = explanation_result.to_dict('index')

                                        if(verbose):
                                            print(explanation_result)
                                            print()
                                            print("eo (y=" + str(y_val) + ")",
                                                explanation_result.sum().item())
                                            print("Exact eo (y=" + str(y_val) + ")",
                                                entry['exact eo (y=' + str(y_val) + ")"])

                                        # explanation_result = fairXplainer.get_top_k_weights_disentangled(
                                        #     k=10, verbose=verbose)
                                        # entry['disentangled influence function (eo) (y=' + str(
                                        #     y_val) + ")"] = explanation_result.to_dict('index')
                                    else:
                                        entry['fairXplainer degenerate case'] = True

                            # Explanation of sufficiency fairness metrics
                            elif(fairness_metric == "suff"):
                                y_predicted = model[i].predict(data[i])
                                y_unique = np.unique(y_predicted)
                                if(not (len(y_unique) == 2 and 1 in y_unique and 0 in y_unique)):
                                    # raise ValueError()
                                    continue
                                # print(data[i])
                                # print(data[i][y_predicted == 1])
                                # print(data_y[i][y_predicted == 1])
                                # print(data_y[i][y_predicted == 1].sum())
                                # print(data[i][y_predicted == 0])
                                # print(data_y[i][y_predicted == 0])
                                # print(data_y[i][y_predicted == 0].sum())
                                for y_val in [0, 1]:
                                    if(verbose):
                                        print("\n", "%"*50)
                                        print("hat(Y) = ", y_val)

                                    if(y_val == 1):
                                        fairXplainer = FairXplainer(
                                            model[i], data[i][y_predicted == 1], sensitive_attributes, label=data_y[i][y_predicted == 1], verbose=verbose)
                                    else:
                                        fairXplainer = FairXplainer(
                                            model[i], data[i][y_predicted == 0], sensitive_attributes, label=data_y[i][y_predicted == 0], verbose=verbose)

                                    fairXplainer.compute(
                                        approach="hdmr", spline_intervals=args.spline_intervals, maxorder=args.maxorder, verbose=verbose, explain_sufficiency_fairness=True)
                                    entry['maxorder'] = args.maxorder
                                    entry['exact suff (y=' + str(y_val) + ")"] = fairXplainer.statistical_parity_dataset(
                                        verbose=verbose)

                                    if(not entry['fairXplainer degenerate case'] and not fairXplainer.is_degenerate_case):

                                        explanation_result = fairXplainer.get_top_k_weights(
                                            k=10)
                                        entry['estimated suff (y=' + str(y_val) +
                                            ")"] = explanation_result.sum().item()
                                        entry['influence function (suff) (y=' + str(
                                            y_val) + ")"] = explanation_result.to_dict('index')

                                        if(verbose):
                                            print(explanation_result)
                                            print()
                                            print("suff (y=" + str(y_val) + ")",
                                                explanation_result.sum().item())
                                            print("Exact suff (y=" + str(y_val) + ")",
                                                entry['exact suff (y=' + str(y_val) + ")"])

                                        # explanation_result = fairXplainer.get_top_k_weights_disentangled(
                                        #     k=10, verbose=verbose)
                                        # # if(verbose):
                                        # #     print(explanation_result)
                                        # #     print()

                                        # entry['disentangled influence function (suff) (y=' + str(
                                        #     y_val) + ")"] = explanation_result.to_dict('index')
                                    else:
                                        entry['fairXplainer degenerate case'] = True

                                # quit()
                            else:
                                raise ValueError((fairness_metric))
                        except Exception as e:
                            print("Error:", e)
                            continue
                    elif(explainer == "shap"):
                        # Explanation of statistical parity using SHAP
                        if(fairness_metric == "sp"):
                            max_group_mask, min_group_mask = row_masking_based_on_sensitive_groups(
                                model[i], data[i], sensitive_attributes)

                            shapExplanation = ShapExplanation()
                            try:
                                # print(data[i].shape)
                                explanation_result, exact_statistical_parity = shapExplanation.compute(
                                    model[i], data[i], max_group_mask, min_group_mask, model_name)
                            except Exception as e:
                                print(e)
                                continue

                            entry['exact statistical parity'] = abs(
                                exact_statistical_parity)
                            entry['estimated statistical parity'] = explanation_result.sum(
                            ).item()
                            entry['influence function (sp)'] = explanation_result.to_dict(
                                'index')

                            if(verbose):
                                print(explanation_result)
                                print()
                                print("Statistical parity:",
                                      explanation_result.sum().item())
                                print("Exact statistical parity:",
                                      entry['exact statistical parity'])
                            
                            if(args.exp_intervention and model_name in ['lr']):
                                import copy
                                fifs = []
                                feature_subsets = []
                                modified_sps = []
                                ordered_feature_list = []
                                for feature_subset in explanation_result.index:
                                    if(feature_subset in ["Residual FIFs", "FIFs $(\lambda > 1)$"]):
                                        continue
                                    ordered_feature_list = feature_subset.split("/")
                                    ordered_feature_list = list(set(ordered_feature_list))
                                    
                                    clf_modified = copy.deepcopy(model[i])
                                    for feature in ordered_feature_list:
                                        clf_modified.coef_[0][list(data[i].columns).index(feature)] = 0
                                    fairXplainer_modified = FairXplainer(clf_modified, data[i], sensitive_attributes)    

                                    fairXplainer_modified.compute(approach = 'hdmr', compute_sp_only=True, verbose=False, spline_intervals=4)
                                    # print(ordered_feature_list, fairXplainer_modified.statistical_parity_sample())
                                    modified_sps.append(fairXplainer_modified.statistical_parity_sample())
                                    feature_subsets.append((" \& ".join(ordered_feature_list)).replace("_", " "))
                                    fifs.append(explanation_result.loc[feature_subset]['weight'].item())
                                
                                entry['fifs'] = list(fifs)
                                entry['modified sps'] = list(entry['exact statistical parity'] - modified_sps)            
                                entry['intervention correlation'], entry['intervention p-value'] = pearsonr(fifs, entry['exact statistical parity'] - modified_sps)
                                entry['cosine similarity'] = np.dot(fifs, entry['exact statistical parity'] - modified_sps) / (np.linalg.norm(fifs) * np.linalg.norm(entry['exact statistical parity'] - modified_sps))

                        # Explanation of equalized odds
                        elif(fairness_metric == "eo"):
                            # print(data[i].shape)
                            y_unique = data_y[i].unique()
                            assert len(
                                y_unique) == 2 and 1 in y_unique and 0 in y_unique
                            for y_val in [0, 1]:
                                if(verbose):
                                    print("\n", "%"*50)
                                    print("Y = ", y_val)

                                if(y_val == 1):

                                    max_group_mask, min_group_mask = row_masking_based_on_sensitive_groups(
                                        model[i], data[i][data_y[i] == 1], sensitive_attributes)

                                    shapExplanation = ShapExplanation()
                                    try:
                                        explanation_result, exact_statistical_parity = shapExplanation.compute(
                                            model[i], data[i][data_y[i] == 1], max_group_mask, min_group_mask, model_name)
                                    except Exception as e:
                                        print(e)
                                        continue

                                else:
                                    max_group_mask, min_group_mask = row_masking_based_on_sensitive_groups(
                                        model[i], data[i][data_y[i] == 0], sensitive_attributes)

                                    shapExplanation = ShapExplanation()
                                    try:
                                        explanation_result, exact_statistical_parity = shapExplanation.compute(
                                            model[i], data[i][data_y[i] == 0], max_group_mask, min_group_mask, model_name)
                                    except Exception as e:
                                        print(e)
                                        continue

                                entry['exact eo (y=' + str(y_val) +
                                      ")"] = exact_statistical_parity
                                entry['estimated eo (y=' + str(y_val) +
                                      ")"] = explanation_result.sum().item()
                                entry['influence function (eo) (y=' + str(
                                    y_val) + ")"] = explanation_result.to_dict('index')

                                if(verbose):
                                    print(explanation_result)
                                    print()
                                    print("eo (y=" + str(y_val) + ")",
                                          explanation_result.sum().item())
                                    print("Exact eo (y=" + str(y_val) + ")",
                                          entry['exact eo (y=' + str(y_val) + ")"])

                        # Explanation of sufficiency fairness metrics
                        elif(fairness_metric == "suff"):
                            continue
                            y_predicted = model[i].predict(data[i])
                            y_unique = np.unique(y_predicted)
                            assert len(
                                y_unique) == 2 and 1 in y_unique and 0 in y_unique
                            for y_val in [0, 1]:
                                if(verbose):
                                    print("\n", "%"*50)
                                    print("hat(Y) = ", y_val)

                                if(y_val == 1):

                                    max_group_mask, min_group_mask = row_masking_based_on_sensitive_groups(
                                        model[i], data[i][y_predicted == 1], sensitive_attributes)

                                    shapExplanation = ShapExplanation()
                                    try:
                                        explanation_result, exact_statistical_parity = shapExplanation.compute(
                                            model[i], data[i][y_predicted == 1], max_group_mask, min_group_mask, model_name)
                                    except Exception as e:
                                        print(e)
                                        continue

                                else:
                                    max_group_mask, min_group_mask = row_masking_based_on_sensitive_groups(
                                        model[i], data[i][y_predicted == 0], sensitive_attributes)

                                    shapExplanation = ShapExplanation()
                                    try:
                                        explanation_result, exact_statistical_parity = shapExplanation.compute(
                                            model[i], data[i][y_predicted == 0], max_group_mask, min_group_mask, model_name)
                                    except Exception as e:
                                        print(e)
                                        continue

                                entry['exact suff (y=' + str(y_val) +
                                      ")"] = exact_statistical_parity
                                entry['estimated suff (y=' + str(y_val) +
                                      ")"] = explanation_result.sum().item()
                                entry['influence function (suff) (y=' + str(
                                    y_val) + ")"] = explanation_result.to_dict('index')

                                if(verbose):
                                    print(explanation_result)
                                    print()
                                    print("suff (y=" + str(y_val) + ")",
                                          explanation_result.sum().item())
                                    print("Exact suff (y=" + str(y_val) + ")",
                                          entry['exact suff (y=' + str(y_val) + ")"])

                        else:
                            raise ValueError(fairness_metric)
                    end_time = time()

                    entry['time'] = end_time - start_time
                    entry['dt_depth'] = depth
                    entry['explainer'] = explainer
                    entry['token'] = token
                    entry['cv index'] = i
                    result = pd.concat(
                        [result, pd.DataFrame([entry])], ignore_index=True)
                    # result = result.append(entry, ignore_index=True)

                    if(verbose):
                        print(entry['time'])
                        from pprint import pprint
                        pprint(entry)
                        print(
                            ", ".join(["\'" + column + "\'" for column in result.columns.tolist()]))
                    os.system("mkdir -p data/output")
                    if(args.exp_sampling):
                        result.to_csv('data/output/result_sampling.csv', header=False,
                                      index=False, mode='a')
                    elif(args.exp_intervention):
                        result.to_csv('data/output/result_intervention.csv', header=False,
                                      index=False, mode='a')
                    else:
                        result.to_csv('data/output/result.csv', header=False,
                                      index=False, mode='a')

                    if(explainer == "fairsquare"):
                        break
