from sklearn.model_selection import RandomizedSearchCV

def get_best_model_parameters( X_train, y_train, xgb):
    """_summary_

    Args:
        
        X_train (_type_): _description_
        y_train (_type_): _description_
        xgb (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    xgb_reg = RandomizedSearchCV(estimator=xgb,param_distributions=model_configs["hyper_params"],
                            n_iter = 1,scoring="neg_root_mean_squared_error",cv = 5)

    xgb_reg = xgb_reg.fit(X_train,y_train)
    best_param = xgb_reg.best_params_
    #TODO: Add logger
    return best_param