import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

import re


def get_discount_from_price_info(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df["discount"] = (df["market_price"] - df["sale_price"]) / df["market_price"] * 100

    return df

def get_recommendations(model_configs, cosine_sim,input_df,indices):
    """_summary_

    Args:
        model_configs (_type_): _description_
        cosine_sim (_type_): _description_
        input_df (_type_): _description_
        indices (_type_): _description_

    Returns:
        _type_: _description_
    """
    idx = indices[model_configs["user_prompt"]]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    rec_indices = [i[0] for i in sim_scores]

    return input_df[model_configs["output_fields"]].iloc[rec_indices]

def get_linear_kernel_similarity(input_df: pd.DataFrame):
    """_summary_

    Args:
        input_df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(input_df['description'])
    
    sim_linear = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(input_df.index, index=input_df['product']).drop_duplicates()
    rmv_spc = lambda a:a.strip()
    get_list = lambda a:list(map(rmv_spc,re.split('& |, |\*|\n', a)))
    get_list('A & B, C')
    for col in ['category', 'sub_category', 'type']:
        input_df[col] = input_df[col].apply(get_list)
    
    return indices,sim_linear,input_df

def cleaner(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def get_cleaned_text(input_df):
    for col in ['category', 'sub_category', 'type','brand']:
        input_df[col] = input_df[col].apply(cleaner)
    return input_df

def couple(x):
    return ' '.join(x['category']) + ' ' + ' '.join(x['sub_category']) + ' '+x['brand']+' ' +' '.join( x['type'])
    
def get_soup_column(input_df):
    input_df['soup'] = input_df.apply(couple, axis=1)
    return input_df

def get_cosine_sim(input_df):
    """_summary_

    Args:
        input_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(input_df['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    input_df = input_df.reset_index()
    indices = pd.Series(input_df.index, index=input_df['product'])
    return input_df,indices,cosine_sim



