from hybrid_recommender.AnalyticsCore.audit.logging import (logger)
import os
import pandas as pd
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