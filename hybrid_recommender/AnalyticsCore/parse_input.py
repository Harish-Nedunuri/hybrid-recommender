import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import os


def load_data(input_table: str) -> pd.DataFrame:
    """_summary_

    Args:
        input_table (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    print(f"Reading input file: {input_table}")
    try:
        data = pd.read_csv(input_table, index_col="index")
        data=data.dropna()
    # TODO define a explicit schema, auto schema on read is not recommended
    # session = SparkSession.builder.appName('First App').getOrCreate()
    # data = session.read.options(header='True', inferSchema='True', delimiter=',') \
    #         .csv(input_data_filename)

    except TypeError:
        print(f"Could NOT load files: {input_table}")
        data = []

    return data


def load_config_json(load_config_json_filename: dict) -> dict:
    """_summary_

    Args:
        load_config_json_filename (dict): _description_

    Returns:
        dict: _description_
    """
    f = open (load_config_json_filename, "r")
    model_parameters = json.load(f)
    return model_parameters


def parse_arguments():

    parser = argparse.ArgumentParser(description="hybrid_recommender")
    parser.add_argument(
        "-c",
        "--config-data",
        default=None,
        type=Path,
        help="Path to input config json file",
    )
    return parser.parse_args()
