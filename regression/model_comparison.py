import pandas as pd
import numpy as np
from itertools import combinations
from typing import Iterator, Literal
from scipy.stats import f
from dataclasses import dataclass
from functools import cached_property
from regression.linear_regression import LinearModel
from regression.model_diagnostics import LinearModelSummary


@dataclass(init=True, frozen=True)
class NestedModelFTest:
    """
    F-Test test between two nested Linear Models
    The reduced_model must be a sub-model of the full_model
    """

    full_model: LinearModel
    reduced_model: LinearModel

    @cached_property
    def sse_diff(self) -> float:
        """
        SSE_f - SSE_r
        (SSE for full model - SSE for reduced model)
        :return:
        """
        return self.reduced_model.sse - self.full_model.sse

    @cached_property
    def df_diff(self) -> float:
        """
        full_model_df - reduced_model_df
        :return:
        """
        return self.reduced_model.df - self.full_model.df

    @cached_property
    def mse_diff(self) -> float:
        """
        sse_diff / df_diff
        :return:
        """
        return self.sse_diff / self.df_diff

    @cached_property
    def f_score(self) -> float:
        """
        F Test Statistic
        """
        return self.mse_diff / self.full_model.mse

    @cached_property
    def p_value(self) -> float:
        """
        P-value
        """
        return float(
            (
                1
                - f.cdf(
                    self.f_score,
                    abs(self.df_diff),
                    self.full_model.df,
                )
            )
        )

    def summary(self, print_summary: bool = True) -> pd.DataFrame:
        """
        get the Summary of the F Test

        :param print_summary: set to true to print the summary (default=True)
        :return:
        """

        df = pd.DataFrame()
        df["Source"] = ["Full Model", "Reduced Model", "Difference"]
        df["Sum Square Error (SSE)"] = [
            self.full_model.sse,
            self.reduced_model.sse,
            self.sse_diff,
        ]
        df["df"] = [self.full_model.df, self.reduced_model.df, self.df_diff]
        df["Mean Square Error (MSE)"] = [
            self.full_model.mse,
            self.reduced_model.mse,
            self.mse_diff,
        ]
        df["F_score"] = [np.nan, np.nan, self.f_score]
        df["p(>F)"] = [np.nan, np.nan, self.p_value]

        if print_summary:
            print(df)
        return df


@dataclass(init=True, frozen=True)
class LinearSubModels:
    full_model: LinearModel

    def sub_model_generator(self) -> Iterator[tuple[tuple[int, ...], LinearModel]]:
        p = self.full_model.predictor_count
        for predictor_groups in [combinations(range(p), r) for r in range(p + 1)]:
            for predictors in predictor_groups:
                yield predictors, self.full_model.get_sub_model(predictors)

    def all_sub_model_summary(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for predictors, model in self.sub_model_generator():
            summary = LinearModelSummary(model).comparison_criterion_summary(
                sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                print_summary=False,
            )
            summary.insert(0, "p", model.predictor_count)
            summary.insert(0, "predictors", str(predictors))
            df = pd.concat([df, summary], ignore_index=True)
        return df

    def unbiased_mallows_search(self) -> pd.DataFrame:
        df = pd.DataFrame()
        predictor_count = 0
        best_predictors: tuple[int, ...] | None = None
        best_model: LinearModel | None = None
        best_score: float | None = None
        for predictors, model in self.sub_model_generator():
            if best_model is not None and len(predictors) > predictor_count:
                summary = LinearModelSummary(best_model).comparison_criterion_summary(
                    sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                    print_summary=False,
                )
                summary.insert(0, "p", best_model.predictor_count)
                summary.insert(0, "predictors", str(best_predictors))
                df = pd.concat([df, summary], ignore_index=True)

                predictor_count = len(predictors)
                best_predictors = None
                best_model = None
                best_score = None

            summary = LinearModelSummary(model).comparison_criterion_summary(
                sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                print_summary=False,
            )
            score = abs(summary["Cp"].iloc[0] - predictor_count - 1)

            if best_score is None or score < best_score:
                best_predictors = predictors
                best_model = model
                best_score = score

        if best_model is not None:
            summary = LinearModelSummary(best_model).comparison_criterion_summary(
                sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                print_summary=False,
            )
            summary.insert(0, "p", best_model.predictor_count)
            summary.insert(0, "predictors", str(best_predictors))
            df = pd.concat([df, summary], ignore_index=True)

        return df

    def forward_selection_step(
        self,
        current_parameters: tuple[int, ...],
        criterion: Literal["R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC"],
    ) -> tuple[int, ...] | None:
        if criterion in ["R_sq", "R_sq_adj"]:
            maximize = True
        elif criterion in ["SSE", "Cp", "AIC", "BIC"]:
            maximize = False
        else:
            raise TypeError(f"Unknown model criterion: {criterion}")

        best_parameters: tuple[int, ...] | None = None
        best_score: float | None = None
        for new_parameter in range(self.full_model.predictor_count):
            if new_parameter in current_parameters:
                continue
            candidate_parameters = current_parameters + (new_parameter,)
            candidate_model = self.full_model.get_sub_model(candidate_parameters)
            candidate_summary = LinearModelSummary(
                candidate_model
            ).comparison_criterion_summary(
                sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                print_summary=False,
            )
            candidate_score = candidate_summary[criterion].iloc[0]
            if (
                (best_score is None)
                or (maximize and candidate_score > best_score)
                or (not maximize and candidate_score < best_score)
            ):
                best_parameters = candidate_parameters
                best_score = candidate_score
        return best_parameters

    def backward_elimination_step(
        self,
        current_parameters: tuple[int, ...],
        criterion: Literal["R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC"],
    ) -> tuple[int, ...] | None:
        if criterion in ["R_sq", "R_sq_adj"]:
            maximize = True
        elif criterion in ["SSE", "Cp", "AIC", "BIC"]:
            maximize = False
        else:
            raise TypeError(f"Unknown model criterion: {criterion}")

        best_parameters: tuple[int, ...] | None = None
        best_score: float | None = None
        for removed_parameter in range(self.full_model.predictor_count):
            if removed_parameter not in current_parameters:
                continue
            temp = list(current_parameters)
            temp.remove(removed_parameter)
            candidate_parameters = tuple(temp)
            candidate_model = self.full_model.get_sub_model(candidate_parameters)
            candidate_summary = LinearModelSummary(
                candidate_model
            ).comparison_criterion_summary(
                sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                print_summary=False,
            )
            candidate_score = candidate_summary[criterion].iloc[0]
            if (
                (best_score is None)
                or (maximize and candidate_score > best_score)
                or (not maximize and candidate_score < best_score)
            ):
                best_parameters = candidate_parameters
                best_score = candidate_score
        return best_parameters

    def forward_selection_summary(
        self,
        criterion: Literal["R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC"],
        stop_early: bool = False,
    ) -> pd.DataFrame:
        if criterion in ["R_sq", "R_sq_adj"]:
            maximize = True
        elif criterion in ["SSE", "Cp", "AIC", "BIC"]:
            maximize = False
        else:
            raise TypeError(f"Unknown model criterion: {criterion}")

        df = pd.DataFrame()
        parameters: tuple[int, ...] = ()
        while len(parameters) < self.full_model.predictor_count:
            # get the current model from the parameters
            current_model = self.full_model.get_sub_model(parameters)
            current_summary = LinearModelSummary(
                current_model
            ).comparison_criterion_summary(
                sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                print_summary=False,
            )
            current_score = current_summary[criterion].iloc[0]

            # find the best parameter to add then find the model and score for the new set of parameters
            next_parameters = self.forward_selection_step(
                current_parameters=parameters, criterion=criterion
            )

            if next_parameters is None:
                break
            next_model = self.full_model.get_sub_model(next_parameters)
            next_summary = LinearModelSummary(next_model).comparison_criterion_summary(
                sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                print_summary=False,
            )
            next_score = next_summary[criterion].iloc[0]

            # if stop_early is True and the next_score doesn't beat the current score. Stop.
            if stop_early and (
                (maximize and next_score < current_score)
                | (not maximize and next_score > current_score)
            ):
                break

            # set our new parameters and add the summary to the summary dataframe
            parameters = next_parameters
            next_summary.insert(0, "p", next_model.predictor_count)
            next_summary.insert(0, "predictors", str(parameters))
            df = pd.concat([df, next_summary], ignore_index=True)
        return df

    def backward_elimination_summary(
        self,
        criterion: Literal["R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC"],
        stop_early: bool = False,
    ) -> pd.DataFrame:
        if criterion in ["R_sq", "R_sq_adj"]:
            maximize = True
        elif criterion in ["SSE", "Cp", "AIC", "BIC"]:
            maximize = False
        else:
            raise TypeError(f"Unknown model criterion: {criterion}")

        parameters: tuple[int, ...] = tuple(range(self.full_model.predictor_count))
        df = LinearModelSummary(self.full_model).comparison_criterion_summary(
            sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
            print_summary=False,
        )
        df.insert(0, "p", len(parameters))
        df.insert(0, "predictors", str(parameters))

        while len(parameters) > 0:
            # get the current model from the parameters
            current_model = self.full_model.get_sub_model(parameters)
            current_summary = LinearModelSummary(
                current_model
            ).comparison_criterion_summary(
                sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                print_summary=False,
            )
            current_score = current_summary[criterion].iloc[0]

            # find the best parameter to remove then find the model and score for the new set of parameters
            next_parameters = self.backward_elimination_step(
                current_parameters=parameters, criterion=criterion
            )

            if next_parameters is None:
                break
            next_model = self.full_model.get_sub_model(next_parameters)
            next_summary = LinearModelSummary(next_model).comparison_criterion_summary(
                sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                print_summary=False,
            )
            next_score = next_summary[criterion].iloc[0]

            # if stop_early is True and the next_score doesn't beat the current score. Stop.
            if stop_early and (
                (maximize and next_score < current_score)
                | (not maximize and next_score > current_score)
            ):
                break

            # set our new parameters and add the summary to the summary dataframe
            parameters = next_parameters
            next_summary.insert(0, "p", next_model.predictor_count)
            next_summary.insert(0, "predictors", str(parameters))
            df = pd.concat([df, next_summary], ignore_index=True)
        return df

    def forward_stepwise_selection(
        self,
        criterion: Literal["R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC"],
        loop_limit: int = 10000,
    ) -> pd.DataFrame:
        if criterion in ["R_sq", "R_sq_adj"]:
            maximize = True
        elif criterion in ["SSE", "Cp", "AIC", "BIC"]:
            maximize = False
        else:
            raise TypeError(f"Unknown model criterion: {criterion}")

        df = pd.DataFrame()
        parameters: tuple[int, ...] = ()
        for loop_count in range(loop_limit):
            print(f"\nstart_loop {loop_count}")
            stop_count = 0

            # get the current model from the parameters
            current_model = self.full_model.get_sub_model(parameters)
            current_summary = LinearModelSummary(
                current_model
            ).comparison_criterion_summary(
                sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                print_summary=False,
            )

            current_score = current_summary[criterion].iloc[0]

            # forward pass

            # find the best parameter to add then find the model and score for the new set of parameters
            next_parameters = self.forward_selection_step(
                current_parameters=parameters, criterion=criterion
            )

            if next_parameters is None:
                next_parameters = parameters
            next_model = self.full_model.get_sub_model(next_parameters)
            next_summary = LinearModelSummary(next_model).comparison_criterion_summary(
                sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                print_summary=False,
            )
            next_score = next_summary[criterion].iloc[0]

            # if the next_score doesn't beat the current score. increment the stop_count and don't add this to the model
            if (maximize and next_score < current_score) | (
                not maximize and next_score > current_score
            ):
                print("dont step forward")
                stop_count += 1
            else:
                # add the summary to the summary dataframe
                next_summary.insert(0, "p", next_model.predictor_count)
                next_summary.insert(0, "predictors", str(next_parameters))
                df = pd.concat([df, next_summary], ignore_index=True)

                # set the next parameters and score
                parameters = next_parameters
                current_score = next_score

            # backwards pass

            # find the best parameter to remove then find the model and score for the new set of parameters
            next_parameters = self.backward_elimination_step(
                current_parameters=parameters, criterion=criterion
            )

            if next_parameters is None:
                next_parameters = parameters
            next_model = self.full_model.get_sub_model(next_parameters)
            next_summary = LinearModelSummary(next_model).comparison_criterion_summary(
                sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                print_summary=False,
            )
            next_score = next_summary[criterion].iloc[0]

            # if the next_score doesn't beat the current score. increment stop_count & don't remove this from the model
            if (maximize and next_score < current_score) | (
                not maximize and next_score > current_score
            ):
                print("don't step back9")
                stop_count += 1
            else:
                # set our new parameters and add the summary to the summary dataframe
                parameters = next_parameters
                next_summary.insert(0, "p", next_model.predictor_count)
                next_summary.insert(0, "predictors", str(parameters))
                df = pd.concat([df, next_summary], ignore_index=True)

            if stop_count == 2:
                break
        return df
