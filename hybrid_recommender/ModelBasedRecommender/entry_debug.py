from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
#model import
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

import datetime
# local imports
from hybrid_recommender.AnalyticsCore.audit.logging import logger
from hybrid_recommender.AnalyticsCore.parse_input import (
    load_config_json,
    load_data,
    parse_arguments,
)
from hybrid_recommender.ModelBasedRecommender.src.transformation import ( 
    get_discount, 
    apply_min_max_scalar,
    transform_df 
)
from hybrid_recommender.ModelBasedRecommender.src.hyper_parameter_tuning import (get_best_model_parameters)


def train_and_publish_recommender(model_configs_filename):
    """_summary_

    Args:
        model_configs_filename (str): _description_

    Returns:
        pd.Dataframe: _description_
    """
    # load the model config
    model_configs = load_config_json(model_configs_filename)

    # load data from source (local csv,json, database table view, delta tabels)
    input_table = os.path.join(model_configs["data_dir"], model_configs["input_table"])   
    input_df = load_data(input_table)
    
    # get discount and transform data
    df = get_discount(input_df) 
    df = transform_df(df)

    # split test train data
    X,y = apply_min_max_scalar(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror')
    best_param = model_configs["best_param"]

    # hyper parameter tuning
    if model_configs["hyper_parameter_tuning"] == True:
        best_param = get_best_model_parameters(X_train, y_train, xgb)

    # model training
    xgb = XGBRegressor(**best_param)
    xgb.fit(X_train,y_train)
    preds = xgb.predict(X_test)
    
    model_performances_current = pd.DataFrame({
            "model":["XGBoost"],
            "model" : [str(cross_val_score(xgb,X,y,cv=5).mean())],
            "mean_absolute_error" : [str(mean_absolute_error(y_test,preds))],
            "root_mean_square_error" : [str(np.sqrt(mean_absolute_error(y_test,preds)))],
            "model_score" : [str(xgb.score(X_test,y_test))],
            "created_at":[datetime.datetime.now()],
            "best_params":[best_param]
            })
    
    output_table = os.path.join(
        model_configs["data_dir"], model_configs["model_performance_table"]
    )
    model_performances_current.to_csv(output_table, mode='a',header=False)
    
    recommender_output = model_performances_current
    print("output: \n", model_performances_current)
    return recommender_output



def main():

    # args = parse_arguments()
    # output = train_and_publish_recommender(args.model_configs)

    model_configs = "/mnt/d/hybrid-recommender/cosmos_db_NoSQL/model_recommender_config.json"

    _ = train_and_publish_recommender(model_configs)
    logger.info("Model Based Recommendation Complete")


if __name__ == "__main__":
    main()
