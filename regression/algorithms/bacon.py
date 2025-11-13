import numpy as np
from scipy.stats import t, chi2
from scipy.spatial import distance
from dataclasses import dataclass
from functools import cached_property
from regression.linear_regression import LinearModel


def find_smallest_n(arr: np.typing.NDArray, n: int) -> list[int]:
    return list(np.argpartition(arr, n)[:n])


@dataclass(init=True, frozen=True)
class BACON:
    """BACON Algorithm for outlier detection"""

    x_data: np.typing.NDArray  # 2D array
    y_data: np.typing.NDArray | None = None  # 1D array

    def __post_init__(self):
        if self.y_data is not None and self.x_data.shape[0] != self.y_data.shape[0]:
            raise Exception(
                f"x_data and y_data need to have the same number of data points:"
                f" X.shape={self.x_data.shape} | Y.shape={self.y_data.shape}"
            )

        if len(self.x_data.shape) > 2:
            raise Exception(
                f"x_data needs to be one or two dimensional: x_data.shape={self.x_data.shape}"
            )

        if self.y_data is not None and len(self.y_data.shape) != 1:
            raise Exception(
                f"y_data needs to be one dimensional: y_data.shape={self.y_data.shape}"
            )

    @cached_property
    def n(self) -> int:
        return self.x_data.shape[0]

    @cached_property
    def x_dim(self) -> int:
        return self.x_data.shape[1]

    @cached_property
    def x_bar(self) -> np.typing.NDArray:
        return np.ones(self.n) @ self.x_data / self.n

    @cached_property
    def linear_model(self) -> LinearModel | None:
        if self.y_data is None:
            return None
        return LinearModel(x_data=self.x_data, y_data=self.y_data)

    def model_subset(self, indexes: list[int]) -> LinearModel:
        """
        fit a linear model on a subset of the full data

        :param indexes: which indexes to use
        :return: the linear Model fit just on the indexes given
        """
        x_in = self.x_data[indexes, :]
        if self.y_data is None:
            raise TypeError("Cannot take linear model without setting y_data")
        else:
            y_in = self.y_data[indexes]
        return LinearModel(x_in, y_in)

    def mahalanobis_distances(
        self, current_basic_subset: list[int] | None = None
    ) -> np.typing.NDArray:
        """
        calculate Mahalanobis Distances between each data point and the mean (x_bar)

        d_i = sqrt((x_i-x_bar).transpose * inv(S_b) * (x_i-x_bar))
        where x_bar is the mean of the x_is
        where S_b is the covariance matrix

        :param current_basic_subset:
            if set, use just these indexes to calculate x_bar and S_b
            otherwise use all data points
        :return: Mahalanobis Distances between each data point and the mean
        """
        if current_basic_subset is None:
            x_data_reduced = self.x_data
        else:
            x_data_reduced = self.x_data[current_basic_subset, :]
        cov_matrix = np.cov(x_data_reduced.transpose())
        cov_matrix_inv = np.linalg.inv(cov_matrix)
        res = []
        for i in range(self.n):
            res.append(
                distance.mahalanobis(self.x_data[i, :], self.x_bar, cov_matrix_inv)
            )
        return np.array(res)

    def t_score_distances(self, current_subset: list[int]) -> np.typing.NDArray:
        """
        calculate the t_score residual distances for each data point
        based on a reduced model

        if the data point is in the reduced model, use the SE for the residuals
        if the data point is not in the reduced model, use the SE for the predicted value

        :param current_subset: indexes for the current set for the reduced model
        :return: t_score_distances
        """
        if self.y_data is None:
            raise TypeError("Cannot find t_score_distance without setting y_data")
        lm_reduced = self.model_subset(indexes=current_subset)
        res = []
        for i in range(self.n):
            x0 = self.x_data[i, :]
            if i in current_subset:
                t_value = (
                    self.y_data[i] - lm_reduced.predict(x0)
                ) / lm_reduced.residual_standard_error(x0)
            else:
                t_value = (
                    self.y_data[i] - lm_reduced.predict(x0)
                ) / lm_reduced.predicted_value_standard_error(x0)
            res.append(t_value)
        return np.array(res)

    def initial_basic_subset(self, m: int) -> list[int]:
        """
        return the initial basic subset for the BACON algorithm
        specifically return the m indexes with the lowest Mahalanobis Distance

        Algorithm 2 Version 1 in the BACON paper

        :param m: how many data points are in the initial basic subset
        :return: indexes for initial basic subset
        """
        return find_smallest_n(arr=self.mahalanobis_distances(), n=m)

    def remove_multivariate_outliers(
        self, m: int, alpha_chi: float, max_iter: int = 10000
    ) -> list[int]:
        """
        identify outliers for Multivariate Data using BACON
        and return the indexes of the data points which are NOT outliers

        Algorithm 3 in the BACON paper

        :param m: how many data points are in the initial basic subset
        :param alpha_chi: significance level for the chi-squared distribution
        :param max_iter: the maximum number of iterations before quitting (default = 10000)
        :return: indexes of non outlier data
        """

        current_subset = self.initial_basic_subset(m)

        n = self.n
        p = self.x_dim

        chi_crit = np.sqrt(chi2.isf(q=alpha_chi / n, df=p))

        for i in range(max_iter):
            previous_subset = current_subset
            distances = self.mahalanobis_distances(previous_subset)

            r = len(current_subset)
            h = (n + p + 1) / 2

            c_np = 1 + (p + 1) / (n - p) + 2 / (n - 1 - 3 * p)
            c_hr = max([0, (h - r) / (h + r)])

            c_npr = c_np + c_hr

            dist_crit = c_npr * chi_crit
            current_subset = np.where(distances < dist_crit)[0].tolist()
            if set(current_subset) == set(previous_subset):
                break
        return current_subset

    def initial_regression_basic_subset(
        self, m: int, alpha_chi: float, max_iter: int = 10000
    ) -> list[int]:
        """
        return the initial basic subset for the BACON algorithm for Linear Regression

        Algorithm 4 in the BACON paper

        :param m: number of data points are in the initial basic subset
        :param alpha_chi: significance level for the chi-squared distribution
        :param max_iter: the maximum number of iterations before quitting (default = 10000)
        :return: indexes of non outlier data
        """
        current_subset = self.remove_multivariate_outliers(
            m=m, alpha_chi=alpha_chi, max_iter=max_iter
        )
        t_values = self.t_score_distances(current_subset=current_subset)
        current_subset = find_smallest_n(t_values, self.x_dim + 2)
        for r in range(len(current_subset), m):
            t_values = self.t_score_distances(current_subset=current_subset)
            current_subset = find_smallest_n(t_values, r + 1)
        return current_subset

    def remove_regression_outliers(
        self,
        m: int,
        alpha_chi: float,
        alpha_t: float | None = None,
        max_iter: int = 10000,
    ) -> list[int]:
        """
        identify outliers for Linear Regression Data using BACON
        and return the indexes of the data points which are NOT outliers

        Algorithm 5 in the BACON paper

        :param m: number of data points are in the initial basic subset
        :param alpha_chi: significance level for the chi-squared distributions
        :param alpha_t: significance level for the t distribution (uses alpha_chi if not set)
        :param max_iter: the maximum number of iterations before quitting (default = 10000)
        :return: indexes of non outlier data
        """
        if alpha_t is None:
            alpha_t = alpha_chi
        current_subset = self.initial_regression_basic_subset(
            m=m, alpha_chi=alpha_chi, max_iter=max_iter
        )
        for i in range(max_iter):
            previous_subset = current_subset
            t_distances = self.t_score_distances(current_subset=previous_subset)
            r = len(previous_subset)
            t_crit = t.isf(q=alpha_t / (2 * (r + 1)), df=r - self.x_dim)
            current_subset = np.where(abs(t_distances) < abs(t_crit))[0].tolist()
            if set(current_subset) == set(previous_subset):
                break
        return current_subset
