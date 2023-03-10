import numpy as np
import matplotlib.pyplot as plt
import heapq
from sklearn.linear_model import LinearRegression


# Different Kernel functions
def rectangular(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t) <= 1)
    res[ind] = 0.5
    return res


def triangular(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t) <= 1)
    res[ind] = 1-np.abs(t[ind])
    return res


def epanechnikov(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t) <= 1)
    res[ind] = 0.75*(1-t[ind]**2)
    return res


def biweight(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t) <= 1)
    res[ind] = (15/16)*(1-t[ind]**2)**2
    return res


def triweight(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t) <= 1)
    res[ind] = (35/32)*(1-t[ind]**2)**3
    return res


def tricube(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t) <= 1)
    res[ind] = (70/81)*(1-np.abs(t[ind])**3)**3
    return res


def gaussian(t):
    res = (1/np.sqrt(2*np.pi))*np.exp(-0.5*t**2)
    return res


def cosine(t):
    res = np.zeros_like(t)
    ind = np.where(np.abs(t) <= 1)
    res[ind] = (np.pi/4)*np.cos(np.pi*t[ind]/2)
    return res


def logistic(t):
    res = 1/(np.exp(t)+2+np.exp(-t))
    return res


def sigmoid(t):
    res = (2/np.pi)/(np.exp(t)+np.exp(-t))
    return res


def silverman(t):
    res = 0.5*np.exp(-np.abs(t)/np.sqrt(2)) * \
        np.sin(np.abs(t)/np.sqrt(2)+np.pi/4)
    return res


# Auxiliary functions for Kernel-smoothing
def mse(Y_hat_train, Y_train, print_results=True):
    """ 
    Print mean squared error for test and train data

    Parameters:
        Y_hat_train: estimated y-values for the training set
        Y_train: true y-values for the training set
        Y_test: true y-values for an independent test set, based on the _same_ x-values as Y_train. 

    Return value:
        tuple(training error, test error)

    """
    train_err = np.mean([abs(yh-y)**2 for y, yh in zip(Y_train, Y_hat_train)])
    # test_err = np.mean([abs(yh-y)**2 for y,yh in zip(Y_test,Y_hat_train)])

    if print_results:
        print("train err: {0:.7f}".format(train_err))
        print("train rmse err: {0:.7f}".format(np.sqrt(train_err)))
        # print("test err: {0:.7f}".format(test_err))
    else:
        return train_err


def kNN(X, Y, x_0, k=20):
    """
    Simple 1-D implementation of kNN average.

    Parameters:
        X: the vector of feature data
        x_0: a particular point in the feature space
        k: number of nearest neighbors to include

    Return value:
        The estimated regression function.

    For our purposes, think of a heapq object as a sorted list with many nice performance properties. 
    The first item is always the smallest. For items that are tuples, the default is to sort
    by the first element in the tuple.
    """
    nearest_neighbors = []
    for x, y in zip(X, Y):
        distance = abs(x-x_0)
        heapq.heappush(nearest_neighbors, (distance, y))
    return np.mean([heapq.heappop(nearest_neighbors)[1] for _ in range(k)])


def kernel_smoother_fast(Y, kernel_weights):
    """
    Generalization of 1-D kNN average, with custom kernel.

    Parameters:
        X: the vector of feature data
        kernel weights

    Return value:
        The estimated regression function at x_0.
    """

    return np.average(Y, weights=kernel_weights)


def kernel_smoother(X, Y, x_0, kernel, width):
    """
    Generalization of 1-D kNN average, with custom kernel.

    Parameters:
        X: the vector of feature data
        x_0: a particular point in the feature space
        kernel: kernel function
        width: kernel width

    Return value:
        The estimated regression function at x_0.
    """
    kernel_weights = [kernel(x_0, x, width) for x in X]

    weighted_average = np.average(Y, weights=kernel_weights)
    return weighted_average


def epanechnikov_kernel(x_0, x, width):
    """
    For a point x_0 in x, return the weight for the given width.
    """
    def D(t):
        if t <= 1:
            # return 3/4*float(1-t*t) <== why doesn't this work?
            return float(1-t*t)*3/4
        else:
            return 0
    return D(abs(x-x_0)/width)


def tri_cube_kernel(x_0, x, width):
    def D(t):
        if t <= 1:
            return float(1-t*t*t)**3
        else:
            return 0
    return D(abs(x-x_0)/width)


def linear_kernel_model(X, Y, x_0, kernel, width):
    """
    1-D kernel-smoothed model with local linear regression.

    Parameters:
        X: the vector of feature data
        x_0: a particular point in the feature space
        kernel: kernel function
        width: kernel width

    Return value:
        The estimated regression function at x_0.
    """
    kernel_weights = [kernel(x_0, x, width) for x in X]

    # the scikit-learn functions want something more numpy-like: an array of arrays
    X = [[x] for x in X]

    wls_model = LinearRegression()
    wls_model.fit(X, Y, kernel_weights)

    B_0 = wls_model.intercept_
    B_1 = wls_model.coef_[0]

    y_hat = B_0 + B_1*x_0

    return y_hat


def kNN_multiD(X, Y, x_0, k=20, kernel_pars=None):
    """
    Simple multi-dimensional implementation of kNN average.

    Parameters:
        X: the vector of feature data
        x_0: a particular point in the feature space
        k: number of nearest neighbors to include

    Return value:
        The estimated regression function at x_0.

    Note: use numpy.linalg.norm for N-dim norms.
    """
    nearest_neighbors = []
    for x, y in zip(X, Y):
        distance = np.linalg.norm(np.array(x)-np.array(x_0))
        heapq.heappush(nearest_neighbors, (distance, y))
    return np.mean([heapq.heappop(nearest_neighbors)[1] for _ in range(k)])


def epanechnikov_kernel_multiD(x_0, x, width=1):
    def D(t):
        #print("width = {}".format(width))
        if t <= 1:
            return float(1-t*t)*3/4
        else:
            return 0
    return D(np.linalg.norm(np.array(x)-np.array(x_0))/width)


def tri_cube_kernel_multiD(x_0, x, width=1):
    def D(t):
        if t <= 1:
            return float(1-t*t*t)**3
        else:
            return 0
    return D(np.linalg.norm(np.array(x)-np.array(x_0))/width)


def generalized_kernel_model_fast(X, Y, x_0, kernel_weights, inds, regressor=LinearRegression):
    """
    Multi-D kernel-smoothed model with local generalized regression.

    Parameters:
        X: the vector of feature data
        x_0: a particular point in the feature space
        kernel_weights: 
        inds: higher weighted kernel
        regressor: regression class - must follow scikit-learn API

    Return value:
        The estimated regression function at x_0.
    """
    # Filter out the datapoints with zero weights.
    # Speeds up regressions with kernels of local support.
    model = regressor()
    model.fit(X[inds], Y[inds], sample_weight=kernel_weights[inds])

    return model.predict([x_0])[0]


def generalized_kernel_model(X, Y, x_0, kernel=epanechnikov_kernel_multiD, kernel_pars={}, regressor=LinearRegression):
    """
    Multi-D kernel-smoothed model with local generalized regression.

    Parameters:
        X: the vector of feature data
        x_0: a particular point in the feature space
        kernel: kernel function
        width: kernel width
        regressor: regression class - must follow scikit-learn API

    Return value:
        The estimated regression function at x_0.
    """
    kernel_weights = [kernel(x_0, x, **kernel_pars) for x in X]
    model = regressor()
    model.fit(X, Y, sample_weight=kernel_weights)

    return model.predict([x_0])[0]


def plot(X, Y, Y_hat, title=""):
    """
    Plot data and estimated regression function

    Parameters:
        X: independant variable
        Y: dependant variable
        Y_hat: estimate of the dependant variable; f_hat(X)
    """
    plt.scatter(X, Y, label='data')
    plt.plot(X, Y_hat, label='estimate', color='g', linewidth=2)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.clf()


def basic_linear_regression(x, y):
    """
    Use least-square minimization to compute the regression coefficients
    for a 1-dim linear model.

    parameters:
        x: array of values for the independant (feature) variable 
        y: array of values for the dependaent (target) variable

    return value:
        2-tuple of slope and y-intercept
    """

    # Basic computations to save a little time.
    length = len(x)
    sum_x = sum(x)
    sum_y = sum(y)

    # Σx^2, and Σxy respectively.
    sum_x_squared = sum(map(lambda a: a * a, x))
    sum_of_products = sum([x[i] * y[i] for i in range(length)])

    # Magic formulae!
    a = (sum_of_products - (sum_x * sum_y) / length) / \
        (sum_x_squared - ((sum_x ** 2) / length))
    b = (sum_y - a * sum_x) / length
    return a, b



def compute_pearson_correlation(X, clf, fairXplainer):
    """For assessing the performance of fairXplainer on linear cla"""
    
    from scipy.stats.stats import pearsonr   
    def get_mask(X, s):
        mask = (True)
        for sensitive_group_str in s.split(", "):
            sensitive_group_str_splitted = sensitive_group_str.split(" = ")
            sensitive_feature = sensitive_group_str_splitted[0]
            value = int(float(sensitive_group_str_splitted[1]))
            # print(sensitive_feature, value)

            mask &= (X[sensitive_feature] == value)
        return X[mask]


    X_max_group = get_mask(
        X, fairXplainer.sensitive_groups[fairXplainer._max_positive_prediction_probability_index])
    prediction_max_group = clf.predict(X_max_group)

    X_min_group = get_mask(
        X, fairXplainer.sensitive_groups[fairXplainer._min_positive_prediction_probability_index])
    prediction_min_group = clf.predict(X_min_group)

    fairXplainer_result = fairXplainer.get_weights()

    synthetic_truth = []
    fif = []

    for i, feature in enumerate(X.columns):
        if(feature not in fairXplainer.sensitive_features):
            # print(feature)
            cov_matrix_max = np.cov(X_max_group[feature] * clf.coef_[0][i], prediction_max_group,
                                    bias=True) / (1 - prediction_max_group.mean())
            cov_matrix_min = np.cov(X_min_group[feature] * clf.coef_[0][i], prediction_min_group,
                                    bias=True) / (1 - prediction_min_group.mean())
            
            synthetic_truth.append((cov_matrix_max - cov_matrix_min)[0, 1])
            fif.append(fairXplainer_result.loc[feature].values[0])
    # prediction_max_group.mean(), prediction_min_group.mean()

    
    return pearsonr(synthetic_truth, fif)



"""
    Code source: https://github.com/meelgroup/justicia
"""
from feature_engine import discretisers as dsc
def get_discretized_df(data, columns_to_discretize=None, verbose=False):
    """ 
    returns train_test_splitted and discretized df
    """

    if(columns_to_discretize is None):
        columns_to_discretize = data.columns.to_list()

    if(verbose):
        print("Applying discretization\nAttribute bins")
    for variable in columns_to_discretize:
        bins = min(4, len(data[variable].unique()))
        if(verbose):
            print(variable, bins)

        # set up the discretisation transformer
        disc = dsc.EqualWidthDiscretiser(bins=bins, variables=[variable])

        # fit the transformer
        disc.fit(data)

        if(verbose):
            print(disc.binner_dict_)

        # transform the data
        data = disc.transform(data)
        if(verbose):
            print(data[variable].unique())

    return data

def get_one_hot_encoded_df(df, columns_to_one_hot, verbose=False):
    """  
    Apply one-hot encoding on categircal df and return the df
    """
    if(verbose):
        print("\n\nApply one-hot encoding on categircal attributes")
    for column in columns_to_one_hot:
        if(column not in df.columns.to_list()):
            if(verbose):
                print(column, " is not considered in classification")
            continue

        # Apply when there are more than two categories or the binary categories are string objects.
        unique_categories = df[column].unique()
        if(len(unique_categories) > 2):
            one_hot = pd.get_dummies(df[column])
            if(verbose):
                print(column, " has more than two unique categories",
                      one_hot.columns.to_list())

            if(len(one_hot.columns) > 1):
                one_hot.columns = [column + "_" +
                                   str(c) for c in one_hot.columns]
            else:
                one_hot.columns = [column for c in one_hot.columns]
            df = df.drop(column, axis=1)
            df = df.join(one_hot)
        else:
            # print(column, unique_categories)
            if(0 in unique_categories and 1 in unique_categories):
                if(verbose):
                    print(column, " has categories 1 and 0")

                continue
            df[column] = df[column].map(
                {unique_categories[0]: 0, unique_categories[1]: 1})
            if(verbose):
                print("Applying following mapping on attribute", column, "=>",
                      unique_categories[0], ":",  0, "|", unique_categories[1], ":", 1)
    if(verbose):
        print("\n")
    return df

