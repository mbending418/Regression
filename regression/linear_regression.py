import numpy as np
import pandas as pd
import math

from typing import Literal
from scipy.stats import norm, t, f
from dataclasses import dataclass
from functools import cached_property


@dataclass(init=True, frozen=True)
class LinearModel:
    """
    Linear Regression model between y_data and x_data

    x_data should be a 2D np.typing.NDArray
    every row of x_data should be a different predictor
    every row of x_data should be a data point

    x_data can be 1D if you only have one predictor

    y_data should be a 1D np.typing.NDArray
    with the same length as there are rows in x_data
    """

    x_data: np.typing.NDArray
    y_data: np.typing.NDArray

    def __post_init__(self):
        if self.x_data.shape[0] != self.y_data.shape[0]:
            raise Exception(
                f"x_data and y_data need to have the same number of data points:"
                f" X.shape={self.x_data.shape} | Y.shape={self.y_data.shape}"
            )

        if len(self.x_data.shape) > 2:
            raise Exception(
                f"x_data needs to be one or two dimensional: x_data.shape={self.x_data.shape}"
            )

        if len(self.y_data.shape) != 1:
            raise Exception(
                f"y_data needs to be one dimensional: y_data.shape={self.y_data.shape}"
            )

    @staticmethod
    def load_from_data_frame(
        df: pd.DataFrame,
        response_variable: str | int = 0,
        predictors: list[str] | list[int] | None = None,
    ) -> "LinearModel":
        """

        :param df: data frame to load from
        :param response_variable: name or index of response variable (default=0)
        :param predictors: names or indexes of predictors
        :return: Linear Model using the DataFrame as data
        """
        if isinstance(response_variable, int):
            y_data = df[df.columns[response_variable]]
            col = str(list(df[df.columns])[response_variable])
        elif isinstance(response_variable, str):
            y_data = df[response_variable]
            col = response_variable
        else:
            raise TypeError(
                f"for response_variable, Expected Type 'int' or 'float', got {type(response_variable)}"
            )

        if predictors is None or len(predictors) == 0:
            predictors = [name for name in list(df.columns) if name != col]

        p_type = type(predictors[0])
        for predictor in predictors:
            if not isinstance(predictor, p_type):
                TypeError(
                    f"All predictors need to be of the same type. Got: {p_type} and {predictor}"
                )

        if isinstance(predictors[0], int):
            predictors = [
                name for index, name in enumerate(df.columns) if index in predictors
            ]

        if isinstance(predictors[0], str):
            x_data = df[predictors]
        else:
            raise TypeError(
                f"for predictors, Expected Type 'List[int]' or 'List[str]', got {type(predictors[0])}"
            )

        return LinearModel(np.array(x_data, np.float64), np.array(y_data, np.float64))

    @staticmethod
    def load_from_csv(
        csv_file: str,
        delimiter: str = ",",
        response_variable: str | int = 0,
        predictors: list[str] | list[int] | None = None,
    ) -> "LinearModel":
        """

        :param csv_file: csv_file to load
        :param delimiter: delimiter for reading csv_file
        :param response_variable: name or index of response variable
        :param predictors: names or indexes of predictors
        :return: LinearModel using the CSV as data
        """
        df = pd.read_csv(csv_file, delimiter=delimiter)
        return LinearModel.load_from_data_frame(
            df, response_variable=response_variable, predictors=predictors
        )

    def get_sub_model(self, predictors: tuple[int, ...]) -> "LinearModel":
        """
        return a linear model with a subset of the predictors

        :param predictors: which predictors to keep
        :return: the submodel
        """
        return LinearModel(self.x_data[:, predictors], self.y_data)

    @cached_property
    def dtype(self) -> np.dtype:
        """
        :return: dtype of the response variable
        """
        return self.y_data.dtype

    @cached_property
    def x_bar(self) -> float:
        """
        :return: average of x_data
        """
        return float(np.average(self.x_data))

    @cached_property
    def y_bar(self) -> float:
        """
        :return: average of y_data
        """
        return float(np.average(self.y_data))

    @cached_property
    def n(self) -> int:
        """
        :return: number of data points
        """
        return self.x_data.shape[0]

    @cached_property
    def predictor_count(self) -> int:
        """
        number of predictor variables
        commonly denoted as "p"
        :return: number of predictors
        """
        if len(self.x_data.shape) == 1:
            return 1
        return self.x_data.shape[1]

    @cached_property
    def parameter_count(self) -> int:
        """
        number of model parameters/coefficients
        commonly demoted as "k"
        :return: number of model parameters/coefficients
        """
        return self.predictor_count + 1

    @cached_property
    def df(self) -> int:
        """
        degrees of freedom
        df = n - p - 1
        :return: degrees of freedom
        """
        return self.n - self.parameter_count

    @cached_property
    def design_matrix(self) -> np.typing.NDArray:
        """
        Design Matrix
        commonly denoted X
        X := [1,x]
        where 1 is a column vector of 1's
        where x is the column vector of x_data
        :return: Design Matrix
        """
        x_data = self.x_data
        if len(x_data.shape) == 1:
            x_data = x_data[..., np.newaxis]
        return np.concat((np.ones((x_data.shape[0], 1)), x_data), axis=1)

    @cached_property
    def c_matrix(self) -> np.typing.NDArray:
        """
        C Matrix
        This is an intermediate matrix used in several calculations
        C_matrix = inverse of design_matrix.transpose()*design_matrix
        :return: Inverse(Design_Matrix.transpose() * Design_Matrix
        """
        return np.linalg.inv(self.design_matrix.transpose() @ self.design_matrix)

    @cached_property
    def projection_matrix(self) -> np.typing.NDArray:
        """
        Projection Matrix also known as Hat Matrix
        Commonly denoted P or H
        this is the matrix such that y_hat = P*y
        P_matrix := X_matrix * C_matrix * X_matrix.transpose()
        :return: Projection Matrix
        """
        return self.design_matrix @ self.c_matrix @ self.design_matrix.transpose()

    @cached_property
    def beta_hat(self) -> np.typing.NDArray:
        """
        estimated model coefficients
        beta_hat := C_matrix * X_matrix.transpose * y_data
        :return: beta_hat
        """
        return self.c_matrix @ self.design_matrix.transpose() @ self.y_data

    @cached_property
    def y_hat(self) -> np.typing.NDArray:
        """
        fitted values for response variable
        y_hat := P*y
        :return: y_hat
        """
        return self.projection_matrix @ self.y_data

    @cached_property
    def residuals(self) -> np.typing.NDArray:
        """
        the residuals of our response variable
        residuals := y_data - y_fitted = y_data - P*y_data
        :return: residuals
        """
        return self.y_data - self.y_hat

    @cached_property
    def sst(self) -> float:
        """
        Sum of Squares Total
        SST := sum((y - y_bar)^2)
        :return: SST
        """
        return float(sum((self.y_data - self.y_bar) ** 2))

    @cached_property
    def ssr(self) -> float:
        """
        Sum of Squares due to Regression
        SSR := sum((y_hat - y_bar)^2) = sum((P*y - y_bar)^2)
        :return: SSR
        """
        return float(sum((self.y_hat - self.y_bar) ** 2))

    @cached_property
    def sse(self) -> float:
        """
        Sum of Squared Errors
        SSE := sum(residuals**2)
        :return: SSE
        """
        return float(sum(self.residuals**2))

    @cached_property
    def mst(self) -> float:
        """
        Mean Sum of Squares Total
        MST := SST / df where df = n - 1
        :return: MST
        """
        return self.sst / (self.n - 1)

    @cached_property
    def msr(self) -> float:
        """
        Mean Sum of Squares due to Regression
        MSR := MSR / df where df = p = predictor_count
        :return: MSR
        """
        return self.ssr / self.predictor_count

    @cached_property
    def mse(self) -> float:
        """
        Mean Sum of Squared Errors
        MSE := SSE / df where df = n-p-1
        :return: MSE
        """
        return self.sse / self.df

    @cached_property
    def r_squared(self) -> float:
        """
        R_squared value (coefficient of determination)
        R_squared := SSR/SST
        :return: R_squared
        """
        return self.ssr / self.sst

    @cached_property
    def r_squared_adjusted(self) -> float:
        """
        Adjusted R_squared value
        R_squared_adjusted := 1 - (SSE / n-k)/(SST/n-1)
        :return: R_squared_adjusted
        """
        return 1 - (self.sse / self.df) / (self.sst / (self.n - 1))

    def mallows_criterion(self, sigma_hat_squared_full_model: float) -> float:
        """
        mallow's criterion for model selection

        Cp = SSE_p/sigma_hat_sq + (2*p - n)

        :param sigma_hat_squared_full_model: the sigma_hat_sq for the full model
        :return: Cp
        """
        return self.sse / sigma_hat_squared_full_model + (
            2 * self.parameter_count - self.n
        )

    @cached_property
    def akaike_information_criterion(self) -> float:
        """
        Akaike Information Criterion (AIC)

        AIC = n * ln(SSE_p/n) + 2*p
        :return: AIC
        """
        return self.n * math.log(self.sse / self.n) + 2 * self.parameter_count

    @cached_property
    def bayes_information_criterion(self) -> float:
        """
        Baye's Information Criterion (BIC)

        BIC = n * ln(SSE_p/n) * p*ln(n)
        :return: BIC
        """
        return self.n * math.log(self.sse / self.n) + self.parameter_count * math.log(
            self.n
        )

    @cached_property
    def correlation(self) -> float:
        """
        correlation between x and y
        r = sqrt(R^squared) with the same sign as B1
        :return: r
        """
        if self.predictor_count == 1:
            return float(math.sqrt(self.r_squared) * np.sign(self.beta_hat[1]))
        else:
            raise TypeError(
                f"Only single predictor models can have correlation: p={self.predictor_count}"
            )

    @cached_property
    def sigma_hat_squared(self) -> float:
        """
        estimate for model noise
        sigma_hat_squared := mse
        :return: sigma_hat_squared
        """
        return self.mse

    @cached_property
    def beta_hat_covariance_matrix(self) -> np.typing.NDArray:
        """
        the variance-covariance matrix for beta_hat
        beta_hat_covariance_matrix = sigma_sq_hat * C
        :return: Cov(beta_hat)
        """
        return self.sigma_hat_squared * self.c_matrix

    @cached_property
    def beta_hat_standard_error(self) -> np.typing.NDArray:
        """
        the standard error for beta_hat
        beta_hat_standard_error[i]= sqrt(beta_hat_covariance_matrix[i][i])
        :return: SE(beta_hat)
        """
        return np.vectorize(math.sqrt)(self.beta_hat_covariance_matrix.diagonal())

    @cached_property
    def beta_hat_t_score(self) -> np.typing.NDArray:
        """
        t-score for the estimated model coefficients
        :return: t_score for beta_hat
        """
        return self.beta_hat / self.beta_hat_standard_error

    @cached_property
    def sse_i(self) -> np.typing.NDArray:
        """
        returns an array of SEE_i values
        each SSE_i is the SEE dropping the ith data point from the calculation:
        :return: SSE_i
        """
        return np.array(
            [sum(np.delete(self.residuals, index) ** 2) for index in range(self.n)],
            self.dtype,
        )

    @cached_property
    def sigma_hat_squared_i(self) -> np.typing.NDArray:
        """
        returns an array of sigma_squared_hat values calculated without the ith data point
        :return: sigma_hat_squared_i
        """
        return self.sse_i / self.df

    @cached_property
    def pii(self) -> np.typing.NDArray:
        """
        returns an array of pii values
        pii = ith entry along the diagonal of the Projection matrix
        :return: pii
        """

        return self.projection_matrix.diagonal()

    @cached_property
    def residuals_internally_standardized(self) -> np.typing.NDArray:
        """
        internally standardized residuals
        ri = ei / (sigma_hat * sqrt(1- pii))
        where ei is the ith residual
        and sigma_hat is the square_root of sigma_hat_squared
        :return: ri
        """
        return np.array(
            [
                self.residuals[index]
                / math.sqrt(self.sigma_hat_squared * (1 - self.pii[index]))
                for index in range(self.n)
            ],
            self.dtype,
        )

    @cached_property
    def residuals_externally_standardized(self) -> np.typing.NDArray:
        """
        externally standardized residuals
        ri_star = ei / (sigma_hat_i * sqrt(1-pii))
        where ei is the ith residuals
        and sigma_hat_i is the square_root of sigma_hat_squared_i
        :return: ri_star
        """
        return np.array(
            [
                self.residuals[index]
                / math.sqrt(self.sigma_hat_squared_i[index] * (1 - self.pii[index]))
                for index in range(self.n)
            ],
            self.dtype,
        )

    @cached_property
    def cooks_distance(self) -> np.typing.NDArray:
        """
        Array of Cook's Distance for each data point
        Ci = ri_squared / (n - df) * pii / (1-pii)
        :return: array of Ci
        """
        return (
            self.residuals_internally_standardized**2
            / (self.n - self.df)
            * self.pii
            / (1 - self.pii)
        )

    @cached_property
    def welsh_and_kuh_measure(self) -> np.typing.NDArray:
        """
        Array of the Welsh and Kuh Measure for each data point
        DFITSi = (externally studentized residuals) * sqrt(pii/(1-pii))
        :return: array of DFITSi
        """
        return self.residuals_externally_standardized * np.sqrt(
            self.pii / (1 - self.pii)
        )

    @cached_property
    def hadis_influence_measure(self) -> np.typing.NDArray:
        """
        Array of the Hadi's Influence Measure for each data point
        Hi = pii/(1-pii) + (p+1)/(1-pii) * di^2/(1-di^2)
        di = normalized residual = ei/sqrt(SSE)
        :return: array of Hi
        """
        di = self.residuals / math.sqrt(self.sse)
        return self.pii / (1 - self.pii) + self.parameter_count / (
            1 - self.pii
        ) * di**2 / (1 - di**2)

    @cached_property
    def theoretical_quantiles(self) -> np.typing.NDArray:
        """
        Array of theoretical quantiles

        Plot against sorted standardized residuals for a QQ-Plot
        which tests that the standardized residuals are normally distributed

        :return: array of theoretical quantiles
        """
        return np.array([norm.ppf((i + 0.5) / self.n) for i in range(0, self.n)])

    @cached_property
    def general_f_score(self) -> float:
        """
        General F Test score for this model
        :return: F-score
        """

        return self.msr / self.mse

    @cached_property
    def general_f_test(self) -> float:
        """
        :return: p_value from the General F Test
        """
        return float((1 - f.cdf(self.general_f_score, self.predictor_count, self.df)))

    def predict(self, x0: float | np.typing.NDArray) -> float:
        """
        returns the predicted value of the model at x0

        x0 : input x-value
        x0 should be an NDArray of predictors the size of predictor_count
        if you only have one predictor x0 can be a float
        :return: fitted model value at x0
        """
        if isinstance(x0, float):
            x0 = np.array([x0])
        if x0.shape != (self.predictor_count,):
            raise Exception(
                f"wrong shape for x0. Expected=({self.predictor_count},) Actual={x0.shape}"
            )
        x0 = np.concat((np.array([1]), x0))
        return float((x0 @ self.beta_hat))

    def predicted_value_standard_error(self, x0: float | np.typing.NDArray) -> float:
        """
        returns the standard error for the predicted value of the model at x0

        :param x0: input x-value
        x0 should be an NDArray of predictors the size of predictor_count
        if you only have one predictor x0 can be a float
        :return: standard error for predicted value
        """
        if isinstance(x0, float):
            x0 = np.array([x0])
        if x0.shape != (self.predictor_count,):
            raise Exception(
                f"wrong shape for x0. Expected=({self.predictor_count},) Actual={x0.shape}"
            )
        x0 = np.concat((np.array([1]), x0))
        return float(math.sqrt(self.sigma_hat_squared * (1 + x0 @ self.c_matrix @ x0)))

    def mean_response_standard_error(self, x0: float | np.typing.NDArray) -> float:
        """
        returns the standard error for the average predicted value of the model at x0

        :param x0: input x-value
        :return: standard error for mean response
        """
        if isinstance(x0, float):
            x0 = np.array([x0])
        x0 = np.concat((np.array([1]), x0))
        return float(math.sqrt(self.sigma_hat_squared * (x0 @ self.c_matrix @ x0)))

    def beta_hat_p_values(
        self,
        offsets: np.typing.NDArray | None = None,
        sided: Literal["one-sided", "two-sided"] = "two-sided",
    ) -> np.typing.NDArray:
        """
        perform a simple t-test on the model parameters

        :param offsets: offset to test against for each model parameter
        :param sided: whether to do a one-sided or two-sided test
        :return: the t-test probabilities for each model parameter
        """
        if offsets is not None:
            t_score = self.beta_hat_t_score - offsets / self.beta_hat_standard_error
        else:
            t_score = self.beta_hat_t_score

        p_value = 1 - t.cdf(abs(t_score), self.df)
        if sided == "two-sided":
            p_value *= 2
        return p_value

    def beta_hat_confidence_interval(self, confidence: float) -> np.typing.NDArray:
        """
        confidence interval on model parameters

        :param confidence: width of confidence interval (e.g. 0.95)
        :return: confidence interval for beta_hat
        """
        t_crit = t.ppf((1 + confidence) / 2, self.df)

        return np.array(
            [
                self.beta_hat - t_crit * self.beta_hat_standard_error,
                self.beta_hat + t_crit * self.beta_hat_standard_error,
            ]
        )

    def prediction_interval(
        self, x0: float | np.typing.NDArray, confidence: float
    ) -> np.typing.NDArray:
        """
        prediction interval around x0

        :param x0: input x-value
        :param confidence: width of confidence interval (e.g. 0.95)
        :return: prediction inteveral around x0
        """
        t_crit = t.ppf((1 + confidence) / 2, self.df)

        return np.array(
            [
                self.predict(x0) - t_crit * self.predicted_value_standard_error(x0),
                self.predict(x0) + t_crit * self.predicted_value_standard_error(x0),
            ]
        )

    def mean_response_confidence_interval(
        self, x0: float | np.typing.NDArray, confidence: float
    ) -> np.typing.NDArray:
        """
        confidence interval around x0

        :param x0: input x-value
        :param confidence: width of confidence interval (e.g. 0.95)
        :return: confidence interval around x0
        """
        t_crit = t.ppf((1 + confidence) / 2, self.df)

        return np.array(
            [
                self.predict(x0) - t_crit * self.mean_response_standard_error(x0),
                self.predict(x0) + t_crit * self.mean_response_standard_error(x0),
            ]
        )
