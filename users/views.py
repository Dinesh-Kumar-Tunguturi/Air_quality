
from django.shortcuts import render, HttpResponse
from django.contrib import messages

from django.http import JsonResponse
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# views.py
from .algorithms.Algorithm import process_data, calculate_mse
from django.http import JsonResponse
from sklearn.ensemble import RandomForestClassifier


# Define your view function for the decision tree model training and metrics calculation

from .algorithms.Algorithm import decision

from sklearn.model_selection import train_test_split

import subprocess

import os

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,roc_auc_score,f1_score

# Create your views here.


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):

    return render(request, 'users/UserHomePage.html', {})


from sklearn.metrics import mean_squared_error

def predict_aqi_bucket(request):
    if request.method == 'POST':
        # Assuming you have a form where users input the feature values
        # For simplicity, I'll assume the form fields are 'pm25', 'pm10', 'no', etc.
        pm25 = float(request.POST['pm25'])
        pm10 = float(request.POST['pm10'])
        no = float(request.POST['no'])
        no2 = float(request.POST['no2'])
        nox = float(request.POST['nox'])
        nh3 = float(request.POST['nh3'])
        co = float(request.POST['co'])
        so2 = float(request.POST['so2'])
        o3 = float(request.POST['o3'])

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'PM2.5': [pm25],
            'PM10': [pm10],
            'NO': [no],
            'NO2': [no2],
            'NOx': [nox],
            'NH3': [nh3],
            'CO': [co],
            'SO2': [so2],
            'O3': [o3]
        })

        # Load the trained model and LabelEncoder
        model, le, X_train, X_test, y_train, y_test = process_data(request)

        # Make predictions on the new data
        X_new = input_data.values
        predicted_bucket = model.predict(X_new)
        predicted_bucket_label = le.inverse_transform(predicted_bucket.astype(int))[0]

        # Calculate the Mean Squared Error on the training data
        y_train_pred = model.predict(X_train)
        mse_train = mean_squared_error(y_train, y_train_pred)

        # Optionally, calculate the Mean Squared Error on the test data
        y_test_pred = model.predict(X_test)
        mse_test = mean_squared_error(y_test, y_test_pred)





        return render(request, 'users/result.html', {
            'predicted_bucket': predicted_bucket_label,
            'mse_train': mse_train,
            'mse_test': mse_test
        })

    return render(request, 'users/prediction_form.html')


import pandas as pd
import seaborn as sns
import plotly.express as px
from django.shortcuts import render

def air_quality_pair_plots(request):
    # Load the air quality data into a DataFrame
    path = settings.MEDIA_ROOT + "//" + 'city_day.csv'
    data = pd.read_csv(path)


    # Assuming 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3' are columns of interest
    columns_of_interest = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']

    # Create a subset of the data with only the columns of interest
    subset_data = data[columns_of_interest]

    # Create pair plots using seaborn
    sns.set(style='ticks')
    pair_plot = sns.pairplot(subset_data)

    # Convert the pair plots to a Plotly figure
    fig = px.imshow(pair_plot.data)

    # Get the plot's HTML representation
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Render the template with the interactive plot
    return render(request, 'users/pair_plots.html', {'plot_html': plot_html})

def decision_tree(request):
    from sklearn import metrics

    path = settings.MEDIA_ROOT + "//" + 'city_day.csv'
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

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    Regressor = DecisionTreeRegressor()
    Regressor.fit(X_train, Y_train)

    prediction = Regressor.predict(X_test)

    mae = metrics.mean_absolute_error(Y_test, prediction)
    mse = metrics.mean_squared_error(Y_test, prediction)
    rmse = np.sqrt(mse)

    return render(request, 'users/dt.html', {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
    })
def lr(request):
    from sklearn import metrics
    from sklearn.linear_model import LinearRegression


    path= settings.MEDIA_ROOT + "//" + 'city_day.csv'
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

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    Regressor = LinearRegression()
    Regressor.fit(X_train, Y_train)

    prediction = Regressor.predict(X_test)

    mae = metrics.mean_absolute_error(Y_test, prediction)
    mse = metrics.mean_squared_error(Y_test, prediction)                      
    rmse = np.sqrt(mse)

    return render(request, 'users/lr.html', {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
    })
def svm(request):
    from sklearn import metrics
    from sklearn.svm import SVR


    path = settings.MEDIA_ROOT + "//" + 'city_day.csv'
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

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    Regressor = SVR()
    Regressor.fit(X_train, Y_train)

    prediction = Regressor.predict(X_test)

    mae = metrics.mean_absolute_error(Y_test, prediction)
    mse = metrics.mean_squared_error(Y_test, prediction)                      
    rmse = np.sqrt(mse)

    return render(request, 'users/svm.html', {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
    })
