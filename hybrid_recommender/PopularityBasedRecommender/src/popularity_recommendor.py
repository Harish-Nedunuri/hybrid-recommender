import pandas as pd
import os
from hybrid_recommender.AnalyticsCore.audit.logging import logger


def popularity_recommendor(df, model_configs, sort_mode=False):
    """_summary_

    Args:
        df (_type_): _description_
        model_configs (_type_): _description_
        col (str, optional): _description_. Defaults to "rating".
        sort_mode (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    rated = df.copy().loc[df["rating"] >= model_configs["rating_threshold"]]
    rated = rated.sort_values(by=model_configs["sort_field"], ascending=sort_mode)

    return rated[model_configs["output_fields"]].head(10)


def publish_recommendation(model_configs, recommender_output):
    """_summary_

    Args:
        model_configs (_type_): _description_
        recommender_output (_type_): _description_
    """
    output_table = os.path.join(
        model_configs["data_dir"], model_configs["output_table"]
    )
    recommender_output.to_csv(output_table)
    logger.info("Popularity output table written to database")
