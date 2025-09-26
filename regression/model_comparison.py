import numpy as np
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

    def __post_init__(self): # TODO: refactor this to be more pythonic
        if not np.array_equal(self.full_model.y_data, self.reduced_model.y_data):
            raise Exception("y_data must match between full model and reduced model")
        full_model_predictors = [self.full_model.x_data[:, i] for i in range(self.full_model.predictor_count)]
        reduced_model_predictors = [self.reduced_model.x_data[:, i] for i in range(self.reduced_model.predictor_count)]
        for index, predictor in enumerate(reduced_model_predictors):
            found = False
            for fm_predictor in full_model_predictors:
                if np.allclose(fm_predictor, predictor):
                    found = True
                    break
            if not found:
                raise Exception(f"every column of reduced_model.x_data has to be in full_model.x_data:"
                                f"column_index={index}")

    @cached_property
    def f_score(self) -> float:
        """
        ANOVA F Test Statistic
        """
        return ((self.reduced_model.sse - self.full_model.sse) / (self.reduced_model.df - self.full_model.df)) / (
                self.full_model.sse / self.full_model.df)

    @cached_property
    def p_value(self) -> float:
        """
        ANOVA P-value
        """
        return float((1 - f.cdf(self.f_score, self.reduced_model.df - self.full_model.df, self.full_model.df)))
