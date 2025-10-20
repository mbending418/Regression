import pandas as pd
import numpy as np
from itertools import combinations
from typing import Iterator, Literal
from scipy.stats import f
from dataclasses import dataclass
from functools import cached_property
from sklearn.model_selection import KFold, ShuffleSplit
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
        :return: SSE_full - SSE_reduced
        """
        return self.reduced_model.sse - self.full_model.sse

    @cached_property
    def df_diff(self) -> float:
        """
        :return: full_model_df - reduced_model_df
        """
        return self.reduced_model.df - self.full_model.df

    @cached_property
    def mse_diff(self) -> float:
        """
        :return: sse_diff/df_diff
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
        F-Test P-value
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
        :return: F Test Summary
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
    """
    A class for doing analysis on SubModels of a Main Linear Model
    """

    full_model: LinearModel

    def sub_model_generator(self) -> Iterator[tuple[tuple[int, ...], LinearModel]]:
        """
        Generator that yields every submodel in order

        Each time the generator is called it yields:
            a tuple of which predictors are in the submodel
            the submodel
        :return: (predictors, submodel)
        """
        p = self.full_model.predictor_count
        for predictor_groups in [combinations(range(p), r) for r in range(p + 1)]:
            for predictors in predictor_groups:
                yield predictors, self.full_model.get_sub_model(predictors)

    def all_sub_model_summary(self) -> pd.DataFrame:
        """
        :return: a pd dataframe summarizing all submodels
        """
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

    def best_sub_model_summary(
        self, *criterion: Literal["R_sq_adj", "Cp", "AIC", "BIC"]
    ) -> pd.DataFrame:
        """
        find the best sub model for each specified criterion
        :return: summary dataframe of best criterion
        """
        df = pd.DataFrame()
        for crit in criterion:
            df[crit] = [None, None]
        for predictors, _ in self.sub_model_generator():
            for crit in criterion:
                if crit in ["R_sq_adj"]:
                    maximize = True
                elif crit in ["Cp", "AIC", "BIC"]:
                    maximize = False
                else:
                    raise TypeError(f"Unknown model criterion: {crit}")

                best_score = df[crit][1]
                current_score = self.get_sub_model_score(
                    predictors=predictors, criterion=crit
                )
                if (
                    best_score is None
                    or (maximize and current_score > best_score)
                    or (not maximize and current_score < best_score)
                ):
                    df[crit] = [predictors, current_score]
        return df

    def get_sub_model_score(
        self,
        predictors: tuple[int, ...],
        criterion: Literal[
            "R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC", "mallows_bias"
        ],
    ) -> float:
        """
        get the score based on 'criterion' for a particular submodel

        :param predictors: which predictors to keep for the submodel
        :param criterion: which criterion to use
            R_sq : Coefficient of Determination
            SSE: Sum Squared Errors
            R_sq_adj: Adjusted R_sq for multilinear regression
            Cp: Mallow's Criterion
            AIC: Akaike Information Criterion
            BIC: Baye's Information Criterion
            mallows_bias: abs difference between Cp and p+1 (p = # predictors)
        :return: sub_model score
        """
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
        """
        find the best sub_model based on 'criterion'

        :param predictors_list: list of tuples where each tuple specifies which predictors to keep
        :param criterion: which criterion to use
            R_sq : Coefficient of Determination (maximize)
            SSE: Sum Squared Errors (minimize)
            R_sq_adj: Adjusted R_sq for multilinear regression (maximize)
            Cp: Mallow's Criterion (minimize)
            AIC: Akaike Information Criterion (minimize)
            BIC: Baye's Information Criterion (minimize)
            mallows_bias: abs difference between Cp and p+1 (minimize)
        :return: which predictors produce the best sub_model
        """
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
        """
        return a list of which predictors to check next
        if forward=True:
            return a list of tuples of predictors where
            each tuple adds one unused predictor
        if forward=False:
            return a list of tuples of predictors where
            each tuple removes one predictor from predictors

        :param predictors: current set of predictors
        :param forward: if you're doing a forward or backward step
        :return: list of tuple of predictors
        """
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
        """
        return a pd DataFrame Summary of a sub_model

        :param predictors: which predictors to use
        :return: the pd DataFrame summary
        """
        sub_model = self.full_model.get_sub_model(predictors)
        summary = LinearModelSummary(sub_model).comparison_criterion_summary(
            self.full_model.sigma_hat_squared, print_summary=False
        )
        summary.insert(0, "p", len(predictors))
        summary.insert(0, "predictors", str(predictors))
        return summary

    def variable_selection_step(
        self,
        criterion: Literal[
            "R_sq", "SSE", "R_sq_adj", "Cp", "AIC", "BIC", "mallows_bias"
        ],
        predictors: tuple[int, ...],
        forward: bool = True,
    ) -> tuple[int, ...]:
        """
        perform a single forward or backward step in variable selection

        1.) based on the current predictors find the list of predictor tuples to check
        (forward = add one predictor, backward = remove one predictor)
        2.0 based on this list, find which one is best and return that

        :param criterion: the criterion to use to determine which is "best"
            R_sq : Coefficient of Determination (maximize)
            SSE: Sum Squared Errors (minimize)
            R_sq_adj: Adjusted R_sq for multilinear regression (maximize)
            Cp: Mallow's Criterion (minimize)
            AIC: Akaike Information Criterion (minimize)
            BIC: Baye's Information Criterion (minimize)
            mallows_bias: abs difference between Cp and p+1 (minimize)
        :param predictors: current tuple of predictors
        :param forward:
            set to True to step forward  (forward selection)
            set to False to step backward (backward elimination)
        :return:
        """
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
        """
        Perform either Forward Selection or Backward Elimination Variable selection
        based on 'criterion'
        Optionally using a step_wise method

        :param selection_method: "forward" or "backward"
            "forward" for Forward Selection
                start at null model
                add predictors at each step
            "backward" for Backward Elimination
            start at full model
            remove predictors at each step
        :param criterion: which criterion determines what model is best
        :param step_wise: Set to True to do a stepwise selection
            if "forward":
                perform a backward elimination step after every forward step
            if "backward":
                perform a forward selection step after every backward step
        :return: pd DataFrame Summary of the variable selection steps
        """
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
        """
        Find the best sub_model for each predictor count based on
        an unbiased Mallow's Criterion
        where a model is considered unbiased if Cp is close to p+1
        :return: pd DataFrame Summary of the best sub_models for each predictor count
        """
        df = pd.DataFrame()
        p = self.full_model.predictor_count
        for predictor_list in [list(combinations(range(p), r)) for r in range(p + 1)]:
            predictors = self.best_sub_model(
                predictors_list=predictor_list, criterion="mallows_bias"
            )
            summary = self.get_sub_model_summary(predictors)
            df = pd.concat([df, summary], ignore_index=True)
        return df


@dataclass(init=True, frozen=True)
class CrossValidation:
    """
    A class for running cross validation on
    model subsets of a base LinearModel
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

    def get_full_model(self, data_in_fold: list[int] | None = None) -> LinearModel:
        """
        return the full model
            if data_in_fold is set, only train the model on the data in the fold

        :param data_in_fold: what data to use for the model
            None (default): use all the data
            tuple[int, ...]: which data points to use for the model
        :return: the full model
        """
        if len(self.x_data.shape) == 1:
            x_data_fold = self.x_data[data_in_fold]
        else:
            x_data_fold = self.x_data[data_in_fold, :]
        y_data_fold = self.y_data[data_in_fold]

        return LinearModel(x_data_fold, y_data_fold)

    def get_models(
        self,
        predictor_list: list[tuple[int, ...]],
        data_in_fold: list[int] | None = None,
    ) -> Iterator[LinearModel]:
        """
        return submodels based on the predictor_list
            if data_in_fold is set, only train the model on the data in the fold

        :param predictor_list: each entry of the list is a tuple of predictors to keep for a sub model
        :param data_in_fold: what data to use for the model
            None (default): use all the data
            tuple[int, ...]: which data points to use for the model
        :return: a generator holding the models
        """

        full_model = self.get_full_model(data_in_fold=data_in_fold)
        return (full_model.get_sub_model(parameters) for parameters in predictor_list)

    def get_model_test_score(
        self,
        predictors: tuple[int, ...],
        training_indicies: list[int],
        testing_indicies: list[int],
    ) -> float:
        """
        return the mspe for this submodel

        :param predictors: each entry of the list is a tuple of predictors to keep for a sub model
        :param training_indicies: a list of indicies for the training set
        :param testing_indicies: a lit of indicies for the test set
        :return: mean_square_prediction_error
        """
        full_model = self.get_full_model(data_in_fold=training_indicies)
        sub_model = full_model.get_sub_model(predictors=predictors)

        y = self.y_data[testing_indicies]
        if len(self.x_data.shape) == 1:
            y_hat = np.array(
                [sub_model.predict(self.x_data[index]) for index in testing_indicies]
            )
        else:
            y_hat = np.array(
                [
                    sub_model.predict(self.x_data[index, list(predictors)])
                    for index in testing_indicies
                ]
            )
        return float(sum((y - y_hat) ** 2) / len(testing_indicies))

    def n_fold_cross_validation(
        self,
        predictor_list: list[tuple[int, ...]],
        n: int,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """
        run an n-fold cross validation on model subsets

        :param predictor_list: each entry of the list is a tuple of parameters to keep for a sub model to test
        :param n: how many folds to divide the data into
        :param random_seed: the random seed to use
        :return: average MSPE (mean square prediction error) of each models
        """
        mspe = np.array([0.0 for predictors in predictor_list], np.float64)

        for train, test in KFold(
            n_splits=n, shuffle=True, random_state=random_seed
        ).split(np.arange(len(self.y_data))):
            mspe += np.array(
                [
                    self.get_model_test_score(
                        predictors=predictors,
                        training_indicies=list(train),
                        testing_indicies=list(test),
                    )
                    for predictors in predictor_list
                ]
            )

        df = pd.DataFrame()
        df[f"predictors"] = predictor_list
        df[f"MSPE average"] = mspe / n
        return df

    def iterative_cross_validation(
        self,
        predictor_list: list[tuple[int, ...]],
        test_size: float,
        number_of_trials: int,
    ) -> pd.DataFrame:
        """
        run an iterative cross validation
            i.e. split the data into train/test
                 train the model on train
                 test the model on test
                 repeat this for each trial and record the average result

        :param predictor_list: each entry of the list is a tuple of parameters to keep for a sub model to test
        :param test_size: how big the test set should be proportionally (between 0 and 1)
        :param number_of_trials: how many times to iterate on cross validation
        :return: average MSPE (mean square prediction error) of each models, SE MSPE of each of the models
        """
        mspe_total = np.array([0.0 for predictors in predictor_list], np.float64)
        mspe_sq_total = np.array([0.0 for predictors in predictor_list], np.float64)

        for train, test in ShuffleSplit(
            n_splits=number_of_trials, test_size=test_size
        ).split(np.arange(len(self.y_data))):
            mspe = np.array(
                [
                    self.get_model_test_score(
                        predictors=predictors,
                        training_indicies=list(train),
                        testing_indicies=list(test),
                    )
                    for predictors in predictor_list
                ]
            )
            mspe_total += mspe
            mspe_sq_total += mspe**2

        mspe_avg = mspe_total / number_of_trials
        mspe_se = (
            mspe_sq_total - number_of_trials * (mspe_avg**2)
        ) ** 0.5 / number_of_trials
        df = pd.DataFrame()
        df = pd.DataFrame()
        df[f"predictors"] = predictor_list
        df[f"MSPE average"] = mspe_avg
        df[f"MSPE std err"] = mspe_se
        return df
