import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score
from xgboost import XGBRegressor


def test_algorithms(df, index, algorithms="classifier", metrics=None, sort_values='f1_score'):
    if isinstance(algorithms, str):
        if algorithms == "classifier":
            algorithms = [LogisticRegression,
                          RidgeClassifier,
                          SGDClassifier,
                          PassiveAggressiveClassifier,
                          KNeighborsClassifier,
                          DecisionTreeClassifier,
                          ExtraTreeClassifier,
                          SVC,
                          GaussianNB,
                          AdaBoostClassifier,
                          BaggingClassifier,
                          RandomForestClassifier,
                          ExtraTreesClassifier,
                          GradientBoostingClassifier
                          ]
        elif algorithms == "regressor":
            algorithms = [
                LinearRegression,
                Lasso,
                Ridge,
                ElasticNet,
                HuberRegressor,
                Lars,
                LassoLars,
                PassiveAggressiveRegressor,
                RANSACRegressor,
                SGDRegressor,
                TheilSenRegressor,
                KNeighborsRegressor,
                DecisionTreeRegressor,
                ExtraTreeRegressor,
                SVR,
                AdaBoostRegressor,
                BaggingRegressor,
                RandomForestRegressor,
                ExtraTreesRegressor,
                GradientBoostingRegressor,
                XGBRegressor]

    result = {"algorithm": [a.__name__ for a in algorithms]}

    if not metrics:
        metrics = [r2_score, mean_squared_error, mean_absolute_error, f1_score]

    for m in metrics:
        for a in algorithms:
            o = a().fit(df.drop(index, axis=1), df[index])
            if m.__name__ not in result:
                result[m.__name__] = []
            result[m.__name__].append(m(df[index], o.predict(df.drop(index, axis=1))))

    if isinstance(sort_values, str):
        sort_values = [sort_values, ]

    result = pd.DataFrame.from_dict(result).sort_values(by=sort_values)

    return result

