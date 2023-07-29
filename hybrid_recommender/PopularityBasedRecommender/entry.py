import numpy as np
import pandas as pd
from pathlib import Path
import os


# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Text Handling Libraries
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# local imports
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
        model_configs_filename (str): _description_

    Returns:
        pd.Dataframe: _description_
    """
    # load data from source (local csv,json, database table view, delta tabels)
    model_configs = load_config_json(model_configs_filename)
    input_table = os.path.join(model_configs["data_dir"], model_configs["input_table"])
    input_df = load_data(input_table)
    input_df = input_df.dropna()
    # acquire recommendation based on Popularity
    recommender_output = popularity_recommendor(input_df, model_configs, sort_mode=True)
    # save recommender output to hive meta store as csv.
    publish_recommendation(model_configs, recommender_output)

    print("output: \n", recommender_output)
    return recommender_output


def main():

    args = parse_arguments()

    _ = train_and_publish_recommender(args.model_configs)
    logger.info("Popularity Recommendation Complete")


if __name__ == "__main__":
    main()
