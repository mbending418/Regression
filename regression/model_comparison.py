from scipy.stats import f
from dataclasses import dataclass
from functools import cached_property
from regression.linear_regression import LinearModel


@dataclass(init=True, frozen=True)
class ANOVA:
    """
    ANOVA test between two nested Linear Models
    The reduced_model must be a sub-model of the full_model
    """

    full_model: LinearModel
    reduced_model: LinearModel

    @cached_property
    def f_score(self) -> float:
        """
        ANOVA F Test Statistic
        """
        return (
            (self.reduced_model.sse - self.full_model.sse)
            / (self.reduced_model.df - self.full_model.df)
        ) / (self.full_model.sse / self.full_model.df)

    @cached_property
    def p_value(self) -> float:
        """
        ANOVA P-value
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
