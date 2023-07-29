import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_discount(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df["discount"] = (df["market_price"] - df["sale_price"]) / df["market_price"] * 100
    return df

def apply_min_max_scalar(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.Dataframe): _description_

    Returns:
        pd.Dataframe: _description_
    """

    df_transform = df.copy()
    columns= df_transform.columns
    df_transform = pd.get_dummies(df_transform,columns=["category","sub_category"],drop_first=True)
    X = df_transform.drop("discount",axis = 1)
    y = df_transform["discount"]
    columns = X.columns
    scaler = MinMaxScaler()
    X[columns] = scaler.fit_transform(X[columns])

    
    return X,y

def transform_df(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df.drop(["description"],axis = 1,inplace=True)
    logSale = np.log(df.sale_price)
    df.sale_price = logSale
    df["rating"].fillna(df.rating.median(),inplace=True)
    df.drop("product",axis = 1,inplace = True)
    df.drop("brand",axis = 1,inplace = True)
    df.drop("type",axis = 1,inplace=True)
    df = df.drop(df[df["market_price"]>1200].index)
    return df

