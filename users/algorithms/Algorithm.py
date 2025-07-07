import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from django.conf import settings
import os

def process_data(request):
    # Load the data
    # path = settings.  + "//" + 'city_day.csv'
    # d = pd.read_csv(path)
    

    path = os.path.join(settings.BASE_DIR, 'media\city_day.csv')
    d = pd.read_csv(path)


    # Your data preprocessing code
    pmean = d["PM2.5"].mean()
    d["PM2.5"].fillna(pmean, inplace=True)
    pmmean = d["PM10"].mean()
    d["PM10"].fillna(pmmean, inplace=True)
    nmean = d["NO"].mean()
    d["NO"].fillna(nmean, inplace=True)
    nomean = d["NO2"].mean()
    d["NO2"].fillna(nomean, inplace=True)
    noxmean = d["NOx"].mean()
    d["NOx"].fillna(noxmean, inplace=True)
    nhmean = d["NH3"].mean()
    d["NH3"].fillna(nhmean, inplace=True)
    cmean = d["CO"].mean()
    d["CO"].fillna(cmean, inplace=True)
    smean = d["SO2"].mean()
    d["SO2"].fillna(smean, inplace=True)
    omean = d["O3"].mean()
    d["O3"].fillna(omean, inplace=True)

    # Drop the 'Benzene' and 'Toluene' columns if they are present in the DataFrame
    if 'Benzene' in d.columns:
        d.drop('Benzene', axis=1, inplace=True)
    if 'Toluene' in d.columns:
        d.drop('Toluene', axis=1, inplace=True)
    if 'Xylene' in d.columns:
        d.drop('Xylene', axis=1, inplace=True)

    le = LabelEncoder()
    d['AQI_Bucket'] = le.fit_transform(d['AQI_Bucket'])


    # Separate features (X) and target variable (y)
    X = d.iloc[:, 2:11].values
    y = d['AQI_Bucket'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=2)


    # Create the Random Forest Classifier and fit the data
    model = RandomForestClassifier()
    model.fit(X, y)

    return model, le, X_train, X_test, y_train, y_test


from sklearn.metrics import mean_squared_error, accuracy_score

def calculate_mse(model, X, y):
    y_pred = model.predict(X)

    # For classification, we can use accuracy as a metric instead of MSE
    accuracy = accuracy_score(y, y_pred)

    return accuracy


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def decision(y_true, y_pred):
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    return mse, mae, rmse

