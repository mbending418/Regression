import pandas as pd
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
    def f_score(self) -> float:
        """
        F Test Statistic
        """
        return (
            (self.reduced_model.sse - self.full_model.sse)
            / (self.reduced_model.df - self.full_model.df)
        ) / (self.full_model.sse / self.full_model.df)

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
                    self.reduced_model.df - self.full_model.df,
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
        df["F_score"] = [self.f_score]
        df["p(F>1)"] = [self.p_value]

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

    def forward_selection_summary(
        self, criterion: Literal["R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC"]
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
            best_parameters = parameters
            best_model = None
            best_summary = None
            best_score = None
            for new_parameter in range(self.full_model.predictor_count):
                if new_parameter in parameters:
                    continue
                candidate_parameters = parameters + (new_parameter,)
                candidate_model = self.full_model.get_sub_model(candidate_parameters)
                candidate_summary = LinearModelSummary(
                    candidate_model
                ).comparison_criterion_summary(
                    sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                    print_summary=False,
                )
                candidate_score = candidate_summary[criterion].iloc[0]
                if (
                    best_score is None
                    or (maximize and candidate_score > best_score)
                    or (not maximize and candidate_score < best_score)
                ):
                    best_parameters = candidate_parameters
                    best_model = candidate_model
                    best_summary = candidate_summary
                    best_score = candidate_score
            if best_summary is None or best_model is None:
                return df
            best_summary.insert(0, "p", best_model.predictor_count)
            best_summary.insert(0, "predictors", str(best_parameters))
            df = pd.concat([df, best_summary], ignore_index=True)
            parameters = best_parameters
        return df

    def backward_elimination_summary(
        self, criterion: Literal["R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC"]
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
            best_parameters = parameters
            best_model = None
            best_summary = None
            best_score = None
            for removed_parameter in range(self.full_model.predictor_count):
                if removed_parameter not in parameters:
                    continue
                candidate_parameters_list: list[int] = list(parameters)
                candidate_parameters_list.remove(removed_parameter)
                candidate_parameters: tuple[int, ...] = tuple(candidate_parameters_list)
                candidate_model = self.full_model.get_sub_model(candidate_parameters)
                candidate_summary = LinearModelSummary(
                    candidate_model
                ).comparison_criterion_summary(
                    sigma_hat_squared_full_model=self.full_model.sigma_hat_squared,
                    print_summary=False,
                )
                candidate_score = candidate_summary[criterion].iloc[0]
                if (
                    best_score is None
                    or (maximize and candidate_score > best_score)
                    or (not maximize and candidate_score < best_score)
                ):
                    best_parameters = candidate_parameters
                    best_model = candidate_model
                    best_summary = candidate_summary
                    best_score = candidate_score
            if best_summary is None or best_model is None:
                return df
            best_summary.insert(0, "p", best_model.predictor_count)
            best_summary.insert(0, "predictors", str(best_parameters))
            df = pd.concat([df, best_summary], ignore_index=True)
            parameters = best_parameters
        return df
