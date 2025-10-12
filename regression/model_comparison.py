import pandas as pd
from scipy.stats import f
from dataclasses import dataclass
from functools import cached_property
from regression.linear_regression import LinearModel


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
