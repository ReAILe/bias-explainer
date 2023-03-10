from sklearn.linear_model import LinearRegression
from itertools import combinations
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import pandas as pd
from tqdm import tqdm
from fairxplainer import utils
from typing import Dict


class KernelSmoothing():
    def __init__(self, problem: Dict):
        assert 'num_vars' in problem
        self.num_vars = problem['num_vars']

        if("names" in problem):
            self.names = problem['names']
        else:
            self.names = None

    def _backfitting(self, X, Y, f_0, start_index, end_index, max_iteration, regressor, verbose):
        for iteration in tqdm(range(max_iteration), disable=True):
            F_prev = self.F.copy()
            for i in range(start_index, end_index):
                if(iteration == 0 and verbose):
                    print("n index:", i)

                Y_prime = Y - f_0 - self.F.sum(axis=1) + self.F[:, i]

                # Backfitting step with smoothed kernel learning

                """
                    1. Local kernel linear model learning
                """
                # Y_estimated = np.array([utils.generalized_kernel_model_fast(X[:, self._component_function_names[i]],
                #                                                             Y_prime,
                #                                                             x_0[self._component_function_names[i]],
                #                                                             self._kernel_matrix[i][j],
                #                                                             self._higher_weight_indices[i][j],
                #                                                             regressor=regressor)
                #                         for j, x_0 in enumerate(X)])

                """
                    2. Nadaraya-Watson
                """

                # Y_estimated = np.array([utils.kernel_smoother_fast(Y_prime,
                #                                             self._kernel_matrix[i][j],)
                #                         for j in range(Y.shape[0])])

                """
                    3. Sklearn Kernel Ridge regression
                """

                krr = KernelRidge(alpha=1.0, kernel="polynomial")
                krr.fit(X[:, self._component_function_names[i]], Y_prime)
                Y_estimated = krr.predict(
                    X[:, self._component_function_names[i]])

                self.F[:, i] = Y_estimated

                # Mean centering
                self.F[:, i] -= Y_estimated.mean()

                if(verbose):
                    print("Y", Y[:5])
                    print("Y_prime", Y_prime[:5])
                    print("Y_estimated", Y_estimated[:5])
                    print("Mean of Y_estimated", Y_estimated.mean())
                    print()

            # check convergence
            max_difference = np.absolute(self.F - F_prev).max()
            if(True):
                print(iteration, max_difference)

            if(max_difference < 0.001):
                break

            # Y_hat = f_0 + F.sum(axis=1)
            # mse(Y_hat, Y)

        if(True):
            print("="*50, "\n")
            Y_hat = f_0 + self.F.sum(axis=1)
            utils.mse(Y_hat, Y)
            print("="*50, "\n")

    def _precomputed_kernels(self, X, kernel, radius, n):
        self._kernel_matrix = []
        self._higher_weight_indices = []
        for i in range(n):
            temp_kernel_weights = []
            temp_higher_weight_indices = []
            for x_0 in X:
                kernel_weights = kernel(np.linalg.norm(
                    X[:, self._component_function_names[i]] - x_0[self._component_function_names[i]], axis=1)/radius)
                temp_kernel_weights.append(kernel_weights)
                temp_higher_weight_indices.append(
                    np.where(np.abs(kernel_weights) > 1e-10)[0])

            self._kernel_matrix.append(temp_kernel_weights)
            self._higher_weight_indices.append(temp_higher_weight_indices)

    def _sensitivity_analysis(self, Y):
        # Sensitivity analysis
        result = []
        V_Y = Y.var()
        names = []
        for i, name in enumerate(self._component_function_names):
            # print(self.F[:, i].shape, Y.shape, np.stack(self.F[:, i], Y))

            C = np.cov(np.stack((self.F[:, i], Y), axis=0))
            # print(C)
            S = C[0, 1] / V_Y

            C = np.cov(
                np.stack((self.F[:, i], self.F.sum(axis=1) - self.F[:, i]), axis=0))
            result.append([C[0, 0], C[0, 1], float(C[0, 0] / V_Y),
                           float(C[0, 1] / V_Y), S])
            # print(S, float(C[0, 0] / V_Y) + float(C[0, 1] / V_Y))
            if(self.names is None):
                names.append("/".join(["X" + str(feature)
                                       for feature in name]))
            else:
                names.append("/".join([self.names[feature_idx]
                                       for feature_idx in name]))

        # print(result)
        df = pd.DataFrame(
            result, columns=['variance', 'covariance', 'Sa', 'Sb', 'S'], index=names)
        # print("Decomposition:", df[['variance', 'covariance']].values.sum(), df['S'].sum())
        # print("Var(Y)", Y.var())
        return df

    def analyze_step_by_step(self, X: np.ndarray, Y: np.ndarray, width: float = 0.35, regressor=LinearRegression, max_order: int = 2, verbose: bool = False, max_iteration: int = 10):

        # Basic checking
        assert self.num_vars == X.shape[1]
        assert X.shape[0] == Y.shape[0]

        n_pts, k = X.shape
        if(max_order > k):
            max_order = k
            print("Max order is too high. Max order is set to", max_order)

        self._component_function_names = []
        n1 = 0
        n2 = 0
        n3 = 0
        for order in range(1, max_order + 1):
            for combination in list(combinations(range(k), order)):
                self._component_function_names.append(list(combination))
                if(order == 1):
                    n1 += 1
                elif(order == 2):
                    n2 += 1
                elif(order == 3):
                    n3 += 1
                else:
                    raise ArgumentError()

        n = len(self._component_function_names)
        self.F = np.zeros((n_pts, n))

        # precompute kernel
        if(True):
            self._precomputed_kernels(
                X, kernel=utils.epanechnikov, radius=width, n=n)

        # Estimation of first order components
        f_0 = Y.mean()
        if(n1 > 0):
            self._backfitting(X, Y, f_0, 0, n1,
                              max_iteration, regressor, verbose)

        # Estimation of second order components
        if(n2 > 0):
            F = self._backfitting(
                X, Y, f_0, n1, n1 + n2, max_iteration, regressor, verbose)

        # Estimation of third order components
        if(n3 > 0):
            F = self._backfitting(X, Y, f_0, n1 + n2, n1 + n2 + n3,
                                  max_iteration, regressor, verbose)

        Y_hat = f_0 + self.F.sum(axis=1)

        return self._sensitivity_analysis(Y=Y)

    def analyze(self, X, Y, width=0.35, regressor=LinearRegression, max_order=2, verbose=False, max_iteration=100):
        # Backfitting algorithm

        n_pts, k = X.shape
        if(max_order > k):
            max_order = k
            print("Max order is too high. Max order is set to", max_order)

        self._component_function_names = []
        for order in range(1, max_order + 1):
            for combination in list(combinations(range(k), order)):
                self._component_function_names.append(list(combination))
        n = len(self._component_function_names)
        self.F = np.zeros((n_pts, n))  # x_0, x_1, (x_0, x_1)

        # precompute kernel
        self._precomputed_kernels(X, kernel=utils.epanechnikov, radius=width)

        # Estimation of all orders together
        f_0 = Y.mean()
        self._backfitting(X, Y, f_0, 0, n,
                          max_iteration, regressor, verbose)

        Y_hat = f_0 + self.F.sum(axis=1)
        utils.mse(Y_hat, Y)

        return self._sensitivity_analysis(Y=Y)
