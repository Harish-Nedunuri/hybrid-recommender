import os
from hybrid_recommender.AnalyticsCore.audit.logging import logger
from hybrid_recommender.AnalyticsCore.parse_input import (
    load_config_json,
    load_data,
    parse_arguments,
)
from hybrid_recommender.PopularityBasedRecommender.src.popularity_recommendor import (
    popularity_recommendor,
    publish_recommendation,
)


def train_and_publish_recommender(model_configs_filename):
    """_summary_

    Args:
        model_configs_filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    # load data from source (local csv,json, database table view, delta tabels)
    model_configs = load_config_json(model_configs_filename)
    input_table = os.path.join(model_configs["data_dir"], model_configs["input_table"])
    input_df = load_data(input_table)
    input_df=input_df.dropna()
    # acquire recommendation based on Popularity
    recommender_output = popularity_recommendor(input_df, model_configs, sort_mode=True)
    # save recommender output to hive meta store as csv.
    publish_recommendation(model_configs, recommender_output)

    print("output: \n", recommender_output)
    return recommender_output


def main():
    # # TODO: Logging here
    # args = parse_arguments()
    # output = train_and_publish_recommender(args.model_configs)
    # logger.info("training completed, model accurancy".format(output))

    model_configs = "/mnt/d/hybrid-recommender/cosmos_db_NoSQL/popularity_recommender_config.json"

    output = train_and_publish_recommender(model_configs)
    logger.info("Popularity Recommendation Complete")


if __name__ == "__main__":
    main()
