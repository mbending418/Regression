import numpy as np
import pandas as pd
from typing import Literal
from matplotlib import pyplot as plt
from dataclasses import dataclass
from regression.linear_regression import LinearModel
from regression.utils.plotting import auto_size_subplots


@dataclass(init=True, frozen=True)
class LinearModelSummary:
    """
    Class for Summarizing a Linear Model
    """

    lm: LinearModel

    def print_full_summary(self):
        """
        print a full model summary

        includes:
        model equation
        coefficient summary
        variance summary
        gof summary
        :return:
        """
        print(self.model_equation())
        print("")
        self.coefficient_summary()
        print("")
        self.anova_summary()
        print("")
        self.comparison_criterion_summary()

    def model_equation(self, digits: int = 4) -> str:
        """
        get the model equation

        :param digits: how many digits to round to
        :return:
        """
        betas = [round(float(beta), digits) for beta in self.lm.beta_hat]
        equation = f"y_hat = {betas[0]}"
        for index, beta in enumerate(betas[1:]):
            equation += f" + {beta}*x_{index}"
        return equation

    def coefficient_summary(
        self, confidence: float | None = None, print_summary: bool = True
    ) -> pd.DataFrame:
        """
        get the coefficient summary

        includes information about:
        beta_hat
        beta_hat_standard_error
        lower_bound (if confidence is set)
        upper_bound (if confidence is set)
        t_score
        p_value
        lower_

        :param confidence: if given, the confidence level for the coefficient confidence interval
                           if not give, don't construct the confidence interval (default)
        :param print_summary: set to True to print the summary (default=True)
        :return:
        """
        df = pd.DataFrame()
        df["Beta_Hat"] = self.lm.beta_hat
        df["Beta_Hat_SE"] = self.lm.beta_hat_standard_error
        if confidence is not None:
            confidence_interval = self.lm.beta_hat_confidence_interval(confidence)
            df[f"lwr ({confidence})"] = confidence_interval[0, :]
            df[f"upr ({confidence})"] = confidence_interval[1, :]
        df["t_score"] = self.lm.beta_hat_t_score
        df["p(>|t|)"] = self.lm.beta_hat_p_values()

        if print_summary:
            print(df)
        return df

    def anova_summary(self, print_summary: bool = True) -> pd.DataFrame:
        """
        get the anova table

        includes the following:
        SSR  SSR_df MSR  F-Test p(F>1)
        SSE  SSE_df MSE
        SST  SST_df MST

        :param print_summary: set to True to print the summary (default=True)
        :return:
        """

        df = pd.DataFrame()
        df["Source"] = ["Regression", "Residuals", "Total"]
        df["Sum of Squares"] = [self.lm.ssr, self.lm.sse, self.lm.sst]
        df["df"] = [self.lm.predictor_count, self.lm.df, self.lm.n - 1]
        df["Mean Square"] = [self.lm.msr, self.lm.mse, self.lm.mst]
        df["F_score"] = [self.lm.general_f_score, np.nan, np.nan]
        df["p(>F)"] = [self.lm.general_f_test, np.nan, np.nan]

        if print_summary:
            print(df)
        return df

    def comparison_criterion_summary(
        self,
        sigma_hat_squared_full_model: float | None = None,
        print_summary: bool = True,
    ) -> pd.DataFrame:
        """
        get the correlation summary

        includes the following:
        correlation coefficient (if single predictor)
        R_sq value
        SSE
        R_sq_adj value
        Cp
        AIC
        BIC

        :param sigma_hat_squared_full_model: the sigma_hat_sq for the full model (optional)
        :param print_summary: set to True to print the summary (default=True)
        :return:
        """
        df = pd.DataFrame()
        if self.lm.predictor_count == 1:
            df["r"] = [self.lm.correlation]
        df["R_Sq"] = [self.lm.r_squared]
        df["SSE"] = [self.lm.sse]
        df["R_Sq_Adj"] = [self.lm.r_squared_adjusted]
        if sigma_hat_squared_full_model is not None:
            df["Cp"] = [self.lm.mallows_criterion(sigma_hat_squared_full_model)]
        df["AIC"] = [self.lm.akaike_information_criterion]
        df["BIC"] = [self.lm.bayes_information_criterion]

        if print_summary:
            print(df)
        return df

    def prediction_confidence_interval(
        self,
        x0: float | np.typing.NDArray,
        confidence: float,
        interval_type: Literal["confidence", "prediction"] = "confidence",
        print_summary: bool = True,
    ) -> pd.DataFrame:
        """
        construct a summary table of a confidence or prediction interval around a fitted value

        :param x0: input x-value to calculate the fitted value from
        :param confidence:  width of confidence interval (e.g. 0.95)
        :param interval_type: "confidence" or "prediction" (default: confidence)
            set to "confidence" for a Confidence Interval (confidence interval of the mean response)
            set to "prediction" for a Prediction Interval (confidence interval of the prediction)
        :param print_summary: set to True to print the summary (default=True)
        :return:
        """
        df = pd.DataFrame()
        df["fit"] = [self.lm.predict(x0)]
        if interval_type == "confidence":
            bounds = self.lm.mean_response_confidence_interval(x0, confidence)
        elif interval_type == "prediction":
            bounds = self.lm.prediction_interval(x0, confidence)
        else:
            raise Exception(f"Unknown Interval Type: {interval_type}")
        df["lower_bound"] = [bounds[0]]
        df["upper_bound"] = [bounds[1]]

        if print_summary:
            print(df)
        return df


@dataclass(init=True, frozen=True)
class LinearModelPlots:
    """
    Class for creating Diagnostic Plots for a Linear Model
    """

    lm: LinearModel

    def simple_linear_regression_plot(
        self,
        show_scatter_plot: bool = True,
        show_regression_line: bool = False,
        show_error_bars: Literal["mean", "predicted"] | None = None,
        confidence_level: float = 0.95,
    ):
        """
        For a Simple Linear Model (one predictor)
        optionally plot the y vs. x scatterplot (default: True)
        optionally plot the fitted regression line
        optionally plot error bars for mean response or predicted response around the regression line

        :param show_scatter_plot: set to True to show the Y vs X Scatter Plot
        :param show_regression_line: set to true to add a regression line
        :param show_error_bars:
            set to None (default) to plot no error bars
            set to "mean" to plot the "mean response confidence interval" around the regression line
            set to "predicted" to plot the "predicted value prediction interval" around the regression line
        :param confidence_level: the confidence level for the error bars
        :return:
        """
        if self.lm.predictor_count > 1:
            raise Exception(
                "Predictor Count for LinearModel must be 1 for a simple linear regression plot. "
                "Try LinearModelPlots.matrix_plot for Multi-Linear Regression"
            )
        legend = []
        if show_scatter_plot:
            plt.scatter(self.lm.x_data, self.lm.y_data)
            legend.append("Data")
        if show_regression_line:
            plt.plot(self.lm.x_data, self.lm.y_hat, color="red")
            legend.append("Regression Line")
        if show_error_bars == "mean":
            x_min = np.amin(self.lm.x_data)
            x_max = np.amax(self.lm.x_data)
            x_step = (x_max / x_min) / 100
            x = np.arange(x_min, x_max + x_step, x_step)
            interval = np.array(
                [
                    self.lm.mean_response_confidence_interval(
                        x0=x0, confidence=confidence_level
                    )
                    for x0 in x
                ]
            )
            plt.plot(x, interval[:, 0], color="green")
            plt.plot(x, interval[:, 1], color="green")
            legend.append("Confidence Interval")
        elif show_error_bars == "predicted":
            x_min = np.amin(self.lm.x_data)
            x_max = np.amax(self.lm.x_data)
            x_step = (x_max / x_min) / 100
            x = np.arange(x_min, x_max + x_step, x_step)
            interval = np.array(
                [
                    self.lm.prediction_interval(x0=x0, confidence=confidence_level)
                    for x0 in x
                ]
            )
            plt.plot(x, interval[:, 0], color="green")
            plt.plot(x, interval[:, 1], color="green")
            legend.append("Prediction Interval")
        plt.title("Scatterplot Y vs. X")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(legend)

    def matrix_plot(self):
        """
        Create the Matrix Plot for a MultiLinear Model

        ie create a grid plotting:
         the response vs each predictor.
         each predictor vs each other predictor.
        :return:
        """
        fig, ax = plt.subplots(self.lm.predictor_count + 1, self.lm.predictor_count + 1)

        for i in range(self.lm.predictor_count):
            if self.lm.predictor_count == 1:
                x_data = self.lm.x_data
            else:
                x_data = self.lm.x_data[:, i]
            ax[0, i + 1].scatter(x_data, self.lm.y_data)
            ax[i + 1, 0].scatter(x_data, self.lm.y_data)

        for i in range(self.lm.predictor_count):
            for j in range(self.lm.predictor_count):
                if i != j:
                    if self.lm.predictor_count == 1:
                        ax[j + 1, i + 1].scatter(self.lm.x_data, self.lm.x_data)
                    else:
                        ax[j + 1, i + 1].scatter(
                            self.lm.x_data[:, i], self.lm.x_data[:, j]
                        )

        font_size = (30 // self.lm.predictor_count) + 1
        ax[0, 0].text(0.4, 0.4, "Y", fontsize=font_size)
        for i in range(self.lm.predictor_count):
            ax[i + 1, i + 1].text(0.3, 0.4, f"X_{i + 1}", fontsize=font_size)

        plt.show()

    def predictor_residual_plot(
        self,
        predictors: list[int] | None = None,
        standardize_residuals: bool = False,
        absolute_value: bool = False,
    ):
        """
        plot the residuals vs predictors

        :param predictors: which predictors (by index) to show
        :param standardize_residuals: set to True to standardize residuals
        :param absolute_value: set to True to take the absolute value of the residuals
        :return:
        """
        if standardize_residuals:
            residuals = self.lm.residuals_internally_standardized
            y_label = "standardized_residuals"
        else:
            residuals = self.lm.residuals
            y_label = "residuals"
        if absolute_value:
            residuals = abs(residuals)
            y_label = f"abs({y_label})"
        if predictors is None:
            predictors = list(range(self.lm.predictor_count))

        plot_size = auto_size_subplots(self.lm.predictor_count)

        fig, ax = plt.subplots(plot_size[1], plot_size[0])
        for plot_index, predictor_index in enumerate(predictors):
            if max(plot_size) == 1:
                ax_i = ax
                ax_i.scatter(self.lm.x_data, residuals)
            else:
                ax_i = ax[plot_index // plot_size[0], plot_index % plot_size[0]]
                ax_i.scatter(self.lm.x_data[:, predictor_index], residuals)
            ax_i.set_title(f"Predictor {predictor_index}")
            ax_i.set_xlabel(f"x_{predictor_index}")
            ax_i.set_ylabel(y_label)
        plt.show()

    def residual_plot(
        self,
        standardize_residuals: bool = False,
        absolute_value: bool = False,
        x_axis: Literal["fitted", "response", "index"] = "fitted",
    ):
        """
        create a Residual Plot

        :param standardize_residuals: set to True to standardize residuals
        :param absolute_value: set to True to take the absolute value of the residuals
        :param x_axis: what to plot on the x-axis
            use "fitted" to plot y_hat
            use "response" to plot y
            use "index" to plot the data index
        :return:
        """
        if standardize_residuals:
            residuals = self.lm.residuals_internally_standardized
            y_label = "standardized_residuals"
        else:
            residuals = self.lm.residuals
            y_label = "residuals"
        if absolute_value:
            residuals = abs(residuals)
            y_label = f"abs({y_label})"
        if x_axis == "fitted":
            x = self.lm.y_hat
            x_label = "fitted values"
        elif x_axis == "response":
            x = self.lm.y_data
            x_label = "response variable"
        elif x_axis == "index":
            x = np.array(range(self.lm.n), np.float64)
            x_label = "data index"
        else:
            raise Exception(
                f"Unknown Option for LinearModelPlots.residual_plot: x_axis={x_axis}"
            )

        plt.scatter(x, residuals)
        plt.title("Residual Plot")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    def scale_location_plot(self):
        """
        Create a Scale Location Plot

        i.e. the sqrt(abs(standardized residuals)) vs fitted values
        :return:
        """
        plt.scatter(
            self.lm.y_hat, np.sqrt(abs(self.lm.residuals_internally_standardized))
        )
        plt.title("Scale Location")
        plt.xlabel("fitted values")
        plt.ylabel("sqrt(abs(standardized residuals))")

    def qq_plot(self):
        """
        Create a QQ-Plot

        i.e. Theoretical Quantiles vs. Observed Quantiles
        Any deviations from the red line can indicate lack of normality in the residuals
        :return:
        """

        quantiles = list(self.lm.residuals_internally_standardized)
        quantiles.sort()
        plt.scatter(self.lm.theoretical_quantiles, quantiles)
        plt.plot(
            self.lm.theoretical_quantiles, self.lm.theoretical_quantiles, color="red"
        )
        plt.title("QQ-Plot")
        plt.xlabel("theoretical quantiles")
        plt.ylabel("observed quantiles")

    def leverage_plot(self):
        """
        Create a Leverage Plot

        i.e. plot Pii vs Index
        :return:
        """

        plt.scatter(range(1, self.lm.n + 1), self.lm.pii)
        plt.title("Leverage Plot")
        plt.xlabel("index")
        plt.ylabel("Pii")

    def cooks_plot(self):
        """
        Create a Plot of Cook's Distance

        i.e. plot Ci vs Index
        :return:
        """
        plt.scatter(range(1, self.lm.n + 1), self.lm.cooks_distance)
        plt.title("Cook's Distance Plot")
        plt.xlabel("index")
        plt.ylabel("Ci")
