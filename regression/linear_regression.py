import numpy as np
import math

from dataclasses import dataclass
from functools import cached_property


@dataclass(init=True, frozen=True)
class LinearModel:
    x_data: np.typing.NDArray
    y_data: np.typing.NDArray

    def __post_init__(self):
        if self.x_data.shape != self.y_data.shape:
            raise Exception(f"x_data and y_data shapes need to be the same:"
                            f" X.shape={self.x_data.shape} | Y.shape={self.y_data.shape}")

        if len(self.x_data.shape) != 1:
            raise Exception(f"x_data needs to be one dimensional: x_data.shape={self.x_data.shape}")

        if len(self.y_data.shape) != 1:
            raise Exception(f"y_data needs to be one dimensional: y_data.shape={self.y_data.shape}")

    @cached_property
    def dtype(self) -> np.dtype:
        return self.y_data.dtype

    @cached_property
    def x_bar(self) -> float:
        """
        average of x_data
        """
        return float(np.average(self.x_data))

    @cached_property
    def y_bar(self) -> float:
        """
        average of y_data
        """
        return float(np.average(self.y_data))

    @cached_property
    def n(self) -> np.typing.NDArray:
        """
        number of data points
        """
        return self.y_data.shape[0]

    @cached_property
    def df(self) -> np.typing.NDArray:
        """
        degrees of freedom
        df = n - 2
        """
        return self.n - 2

    @cached_property
    def X_matrix(self) -> np.typing.NDArray:
        """
        Design Matrix, defined as follows:
        X_matrix = [1,x]
        where 1 is a column vector of 1's
        where x is the column vector of x_data
        """
        return np.concat([[np.ones(len(self.x_data))], [self.x_data]]).transpose()

    @cached_property
    def C_matrix(self) -> np.typing.NDArray:
        """
        C Matrix
        This is an intermediate matrix used in several calculations
        C_matrix = inverse of X_matrix.transpose()*X_matrix
        """
        return np.linalg.inv(self.X_matrix.transpose() @ self.X_matrix)

    @cached_property
    def P_matrix(self) -> np.typing.NDArray:
        """
        Projection Matrix
        this is the matrix such that y_hat = P*y
        P_matrix := X_matrix * C_matrix * X_matrix.transpose()
        """
        return self.X_matrix @ self.C_matrix @ self.X_matrix.transpose()

    @cached_property
    def beta_hat(self) -> np.typing.NDArray:
        """
        estimated model coefficients
        beta_hat := C_matrix * X_matrix.transpose * y_data
        """
        return self.C_matrix @ self.X_matrix.transpose() @ self.y_data

    @cached_property
    def y_hat(self) -> np.typing.NDArray:
        """
        fitted values for response variable
        y_hat := P*y
        """
        return self.P_matrix @ self.y_data

    @cached_property
    def residuals(self) -> np.typing.NDArray:
        """
        the residuals of our response variable
        residuals := y_data - y_fitted = y_data - P*y_data
        """
        return self.y_data - self.y_hat

    @cached_property
    def SST(self) -> float:
        """
        Sum of Squares Total
        SST := sum((y - y_bar)^2)
        :return:
        """
        return sum((self.y_data - self.y_bar) ** 2)

    @cached_property
    def SSR(self) -> float:
        """
        Sum of Squares due to Regression
        SSR := sum((y_hat - y_bar)^2) = sum((P*y - y_bar)^2)
        """
        return sum((self.y_hat - self.y_bar) ** 2)

    @cached_property
    def SSE(self) -> float:
        """
        Sum of Squared errors
        SSE := sum(residuals**2)
        """
        return sum(self.residuals**2)

    @cached_property
    def r_squared(self) -> float:
        """
        R_squared value
        R_squared := SSR/SST
        """
        return self.SSR / self.SST

    @cached_property
    def correlation(self) -> float:
        """
        correlation between x and y
        r = sqrt(R^squared) with the same sign as B1
        """
        return float(math.sqrt(self.r_squared) * np.sign(self.beta_hat[1]))

    @cached_property
    def sigma_hat_squared(self) -> np.typing.NDArray:
        """
        estimate for model noise
        sigma_hat_squared := sum(residuals**2) / df = SSE/df
        """
        return self.SSE / self.df

    @cached_property
    def beta_hat_varcov_matrix(self) -> np.typing.NDArray:
        """
        the variance-covariance matrix for beta_hat
        beta_hat_varcov_matrix = sigma_sq_hat * C
        """
        return self.sigma_hat_squared * self.C_matrix

    @cached_property
    def beta_hat_standard_error(self) -> np.typing.NDArray:
        """
        the standard error for beta_hat
        beta_hat_standard_error[i]= sqrt(beta_hat_varcov_matrix[i][i])
        """
        return np.vectorize(math.sqrt)(self.beta_hat_varcov_matrix.diagonal())

    @cached_property
    def SSE_i(self) -> np.typing.NDArray:
        """
        returns an array of SEE_i values
        each SSE_i is the SEE dropping the ith data point from the calculation:
        """
        return np.array([sum(np.delete(self.residuals, index)**2) for index in range(self.n)], self.dtype)

    @cached_property
    def sigma_hat_squared_i(self) -> np.typing.NDArray:
        """
        returns an array of sigma_squared_hat values calculated without the ith data point
        """
        return self.SSE_i / self.df

    @cached_property
    def pii(self) -> np.typing.NDArray:
        """
        returns an array of pii values
        pii = ith entry along the diagonal of the Projection matrix
        """

        return self.P_matrix.diagonal()

    @cached_property
    def residuals_internally_standardized(self) -> np.typing.NDArray:
        """
        internally standardized residuals
        ri = ei / (sigma_hat * sqrt(1- pii))
        where ei is the ith residual
        and sigma_hat is the square_root of sigma_hat_squared
        """
        return np.array([self.residuals[index] /
                         math.sqrt(self.sigma_hat_squared * (1 - self.pii[index]))
                         for index in range(self.n)], self.dtype)

    @cached_property
    def residuals_externally_standardized(self) -> np.typing.NDArray:
        """
        externally standardized residuals
        ri_star = ei / (sigma_hat_i * sqrt(1-pii))
        where ei is the ith residuals
        and sigma_hat_i is the square_root of sigma_hat_squared_i
        """
        return np.array([self.residuals[index] /
                         math.sqrt(self.sigma_hat_squared_i[index] * (1 - self.pii[index]))
                         for index in range(self.n)], self.dtype)

    @cached_property
    def cooks_distance(self) -> np.typing.NDArray:
        """
        Array of Cook's Distance for each data point
        Ci = ri_squared / (n - df) * pii / (1-pii)
        """
        return self.residuals_internally_standardized**2 / (self.n - self.df) * self.pii / (1 - self.pii)

    def predict(self, x0) -> float:
        """
        returns the predicted value of the model at x0

        x0 : input x-value
        :return: fitted model value at x0
        """
        x0 = np.array([1, x0], dtype=np.float64)
        return float((x0 @ self.beta_hat))

    def predicted_value_standard_error(self, x0) -> float:
        """
        returns the standard error for the predicted value of the model at x0

        :param x0: input x-value
        :return: standard error
        """
        x0 = np.array([1, x0], dtype=np.float64)
        return float(math.sqrt(self.sigma_hat_squared * (1 + x0 @ self.C_matrix @ x0)))

    def mean_response_standard_error(self, x0) -> float:
        """
        returns the standard error for the average predicted value of the model at x0

        :param x0: input x-value
        :return: standard error
        """
        x0 = np.array([1, x0], dtype=np.float64)
        return float(math.sqrt(self.sigma_hat_squared * (x0 @ self.C_matrix @ x0)))
