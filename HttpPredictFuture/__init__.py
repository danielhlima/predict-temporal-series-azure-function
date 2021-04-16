import logging

import azure.functions as func
import numpy as np
import joblib
from sklearn import preprocessing
import pandas as pd




def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    prophet_model = joblib.load('model_desafio4_fbprophet.pkl')

    j_data = req.get_json()['data']
    d = {'ds':[j_data]}
    df = pd.DataFrame.from_dict(d)
    df['ds'] = pd.DatetimeIndex(df['ds'])
    y_pred = prophet_model.predict(df)
    yhat = y_pred.iloc[:,[0,15]].copy(deep=True)
    list_y_hat = yhat['yhat'].tolist()
    return func.HttpResponse(str(list_y_hat))