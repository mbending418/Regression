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

    def get_sub_model_score(
        self,
        predictors: tuple[int, ...],
        criterion: Literal[
            "R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC", "mallows_bias"
        ],
    ) -> float:
        sub_model = self.full_model.get_sub_model(predictors)
        if criterion == "R_sq":
            return sub_model.r_squared
        elif criterion == "SSE":
            return sub_model.sse
        elif criterion == "R_sq_adj":
            return sub_model.r_squared_adjusted
        elif criterion == "Cp":
            return sub_model.mallows_criterion(self.full_model.sigma_hat_squared)
        elif criterion == "AIC":
            return sub_model.akaike_information_criterion
        elif criterion == "BIC":
            return sub_model.bayes_information_criterion
        elif criterion == "mallows_bias":
            return abs(
                sub_model.mallows_criterion(self.full_model.sigma_hat_squared)
                - len(predictors)
                - 1
            )
        else:
            raise TypeError(f"Unknown Criterion: {criterion}")

    def best_sub_model(
        self,
        predictors_list: list[tuple[int, ...]],
        criterion: Literal[
            "R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC", "mallows_bias"
        ],
    ) -> tuple[int, ...]:
        if criterion in ["R_sq", "R_sq_adj"]:
            maximize = True
        elif criterion in ["SSE", "Cp", "AIC", "BIC", "mallows_bias"]:
            maximize = False
        else:
            raise TypeError(f"Unknown model criterion: {criterion}")

        best_predictors = predictors_list[0]
        best_score = self.get_sub_model_score(
            predictors=best_predictors, criterion=criterion
        )

        for predictors in predictors_list[1:]:
            score = self.get_sub_model_score(predictors=predictors, criterion=criterion)
            if (maximize and score > best_score) or (
                not maximize and score < best_score
            ):
                best_predictors = predictors
                best_score = score

        return best_predictors

    def get_new_parameters_candidate_list(
        self, predictors: tuple[int, ...], forward: bool = True
    ) -> list[tuple[int, ...]]:
        if forward:
            return [
                predictors + (param,)
                for param in range(self.full_model.predictor_count)
                if param not in predictors
            ]
        else:
            candidate_list = []
            for param in predictors:
                temp = list(predictors)
                temp.remove(param)
                candidate_list.append(tuple(temp))
            return candidate_list

    def get_sub_model_summary(self, predictors: tuple[int, ...]) -> pd.DataFrame:
        sub_model = self.full_model.get_sub_model(predictors)
        summary = LinearModelSummary(sub_model).comparison_criterion_summary(
            self.full_model.sigma_hat_squared, print_summary=False
        )
        summary.insert(0, "p", len(predictors))
        summary.insert(0, "predictors", str(predictors))
        return summary

    def variable_selection_step(
        self,
        criterion: Literal["R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC"],
        predictors: tuple[int, ...],
        forward: bool = True,
    ) -> tuple[int, ...]:
        candidate_list = self.get_new_parameters_candidate_list(
            predictors=predictors, forward=forward
        )
        return self.best_sub_model(predictors_list=candidate_list, criterion=criterion)

    def variable_selection_procedure(
        self,
        selection_method: Literal["forward", "backward"],
        criterion: Literal["R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC"],
        step_wise: bool = False,
    ):
        df = pd.DataFrame()

        if selection_method == "forward":
            predictors: tuple[int, ...] = ()
        elif selection_method == "backward":
            predictors = tuple(range(self.full_model.predictor_count))
        else:
            raise TypeError(f"Unknown Selection Method: {selection_method}")

        while (
            selection_method == "forward"
            and len(predictors) < self.full_model.predictor_count
        ) or (selection_method == "backward" and len(predictors) > 0):
            new_predictors = self.variable_selection_step(
                criterion=criterion,
                predictors=predictors,
                forward=selection_method == "forward",
            )
            if new_predictors == self.best_sub_model(
                predictors_list=[predictors, new_predictors], criterion=criterion
            ):
                predictors = new_predictors
                summary = self.get_sub_model_summary(predictors)
                df = pd.concat([df, summary], ignore_index=True)
            else:
                return df

            if step_wise:
                new_predictors = self.variable_selection_step(
                    criterion=criterion,
                    predictors=predictors,
                    forward=selection_method != "forward",
                )
                if new_predictors == self.best_sub_model(
                    predictors_list=[predictors, new_predictors], criterion=criterion
                ):
                    predictors = new_predictors
                    summary = self.get_sub_model_summary(predictors)
                    df = pd.concat([df, summary], ignore_index=True)

        return df

    def unbiased_mallows_selection(self) -> pd.DataFrame:
        df = pd.DataFrame()
        p = self.full_model.predictor_count
        for predictor_list in [list(combinations(range(p), r)) for r in range(p + 1)]:
            predictors = self.best_sub_model(
                predictors_list=predictor_list, criterion="mallows_bias"
            )
            summary = self.get_sub_model_summary(predictors)
            df = pd.concat([df, summary], ignore_index=True)
        return df
