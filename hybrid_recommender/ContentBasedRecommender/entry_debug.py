import numpy as np
import pandas as pd
from hybrid_recommender.AnalyticsCore.parse_input import (
    load_config_json,
    load_data,
    parse_arguments
)
import numpy as np
import pandas as pd
import os
import datetime

from hybrid_recommender.AnalyticsCore.audit.logging import (logger)
from hybrid_recommender.ContentBasedRecommender.src.transform_input import ( 
    get_discount_from_price_info, 
    get_soup_column,
    get_recommendations,     
    get_cleaned_text,
    get_cosine_sim
    )
from hybrid_recommender.ContentBasedRecommender.src.publish_recommendation import(publish_recommendation)

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

    # get the discount information
    input_df = get_discount_from_price_info(input_df)

    # converting everything to lowercase and also removing spaces between words.
    input_df = get_cleaned_text(input_df) 
    input_df = get_soup_column(input_df)

    #compute the Cosine Similarity Score
    input_df, indices, cosine_sim = get_cosine_sim(input_df)

    # get recommendation based on the cosine similarity score mapping
    recommender_output = get_recommendations(model_configs,cosine_sim,input_df,indices)

    # publish recommendation to db
    publish_recommendation(model_configs,recommender_output)

    # print output
    print("user_prompt: ",model_configs["user_prompt"])
    print("output: \n", recommender_output)
    return recommender_output


def main():

    # args = parse_arguments()
    # output = train_and_publish_recommender(args.model_configs)

    model_configs = "/mnt/d/hybrid-recommender/cosmos_db_NoSQL/content_recommender_config.json"

    _ = train_and_publish_recommender(model_configs)
    logger.info("Content Based Recommendation Complete")


if __name__ == "__main__":
    main()
