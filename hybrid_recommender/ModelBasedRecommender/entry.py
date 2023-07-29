from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import train_test_split

#model import
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn import ensemble
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV

# local imports
from hybrid_recommender.AnalyticsCore.audit.logging import logger
from hybrid_recommender.AnalyticsCore.parse_input import (
    load_config_json,
    load_data,
    parse_arguments,
)
from hybrid_recommender.ModelBasedRecommender.src.transformation import (get_discount, apply_min_max_scalar)
from hybrid_recommender.ModelBasedRecommender.src.training import *
from hybrid_recommender.ModelBasedRecommender.src.publish_model import *
from hybrid_recommender.ModelBasedRecommender.src.inference import *


def train_and_publish_recommender(model_configs_filename):
    """_summary_

    Args:
        model_configs_filename (str): _description_

    Returns:
        pd.Dataframe: _description_
    """
    # load data from source (local csv,json, database table view, delta tabels)
    model_configs = load_config_json(model_configs_filename)
    input_df = load_data(model_configs)
    # acquire recommendation based on Popularity
    df = get_discount(input_df)
    # save recommender output to hive meta store as csv.
    X,y = apply_min_max_scalar(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror')
    param_lst = {
    "learning_rate" : [0.1,0.1,0.15,0.3,0.5],
    "n_estimators" : [100,500,1000,2000,3000],
    "max_depth" : [3,6,9],
    "min_child_weight" : [1,5,10,20],
    "reg_alpha" : [0.001,0.01,0.1],
    "reg_lambda" : [0.001,0.01,0.1]
    }
    xgb_reg = RandomizedSearchCV(estimator=xgb,param_distributions=param_lst,
                            n_iter = 5,scoring="neg_root_mean_squared_error",cv = 5)

    xgb_reg = xgb_reg.fit(X_train,y_train,)

    best_param = xgb_reg.best_params_

    xgb = XGBRegressor(**best_param)
    recommender_output= df
    print("output: \n", recommender_output)
    return recommender_output


def main():

    # args = parse_arguments()
    # output = train_and_publish_recommender(args.model_configs)

    model_configs = "/mnt/d/product_recommender/hybrid_recommender/PopularityBasedRecommender/recommender_config.json"

    output = train_and_publish_recommender(model_configs)
    logger.info("Model Based Recommendation Complete")


if __name__ == "__main__":
    main()
