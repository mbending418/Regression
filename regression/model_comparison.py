from scipy.stats import f
from regression.linear_regression import LinearModel


def anova_f_score(full_model: LinearModel, reduced_model: LinearModel):
    """
    Finds the F test statistic for an ANOVA test between two nested Linear Models
    :param full_model: the full linear model
    :param reduced_model: the reduced model (must be a subset of the full model)
    :return: F Score
    """
    return ((reduced_model.sse - full_model.sse) / (reduced_model.df - full_model.df)) / (
            full_model.sse / full_model.df)


def anova_p_value(full_model: LinearModel, reduced_model: LinearModel):
    """
    Runs an ANOVA test between two nested Linear Models and returns the p_value
    :param full_model:
    :param reduced_model:
    :return: ANOVA P Value
    """
    return float((1 - f.cdf(anova_f_score(full_model, reduced_model), reduced_model.df - full_model.df, full_model.df)))
