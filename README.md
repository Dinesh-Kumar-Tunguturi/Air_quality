# 🌫️ Air Quality Prediction using Machine Learning

This project uses machine learning techniques to **predict air quality** based on historical environmental data. The model takes inputs like PM2.5, PM10, NO2, SO2, CO, and O3 levels and predicts the **Air Quality Index (AQI) category** using trained ML models.

---

## 🚀 Features

- Predicts AQI levels based on pollutant concentrations
- Django web interface for input & display
- Graphs generated using Matplotlib, Seaborn, and Plotly
- Clean and modular code structure
- Trained with real-world city data

---

## 🧠 Tech Stack

| Layer        | Tools Used                            |
|-------------|----------------------------------------|
| Backend      | Python, Django                         |
| Machine Learning | scikit-learn, pandas, joblib       |
| Visualization | matplotlib, seaborn, plotly           |
| Frontend     | HTML, CSS (Bootstrap), JS              |
| Dataset      | `city_day.csv` from real AQI data      |

---

## 📁 Project Structure
Air_quality_prediction/
├── Air_quality/
│ ├── init.py
│ ├── settings.py
│ ├── urls.py
│ └── wsgi.py
├── users/
│ ├── views.py
│ ├── models.py
│ ├── urls.py
├── templates/
│ └── predict.html
├── static/
│ └── assets/
├── city_day.csv
├── model.pkl
├── manage.py
├── README.md


---

## 📸 Sample Output Screenshots

### 🎯 Prediction Page
![Prediction Page](images/prediction_result.png)

### 📊 AQI Bar Chart
![AQI Chart](images/air_pollution_chart.png)

*(Ensure you save your actual screenshots in the `/images` folder for this to work)*

---

## ⚙️ How to Run the Project
python -m venv venv
.\venv\Scripts\activate

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/yourusername/air-quality-prediction.git
cd air-quality-prediction

pip install -r requirements.txt
pip install pandas scikit-learn seaborn matplotlib plotly django joblib
python manage.py runserver

### 📊 Dataset Overview
The dataset city_day.csv contains:

Pollutants: PM2.5, PM10, NO2, CO, SO2, O3

AQI Category (Good, Moderate, Poor, etc.)

City and Date

## 📌 Sample Preprocessing Code

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('city_day.csv')
df.fillna(df.mean(), inplace=True)
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]
y = df['AQI_Category']

model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, 'model.pkl')

## ✅ Sample Prediction
import joblib
model = joblib.load('model.pkl')
result = model.predict([[80, 110, 25, 7, 0.5, 30]])
print("Predicted AQI Category:", result[0])
## 📦 Dependencies (requirements.txt)

Django>=4.0
pandas>=1.3
scikit-learn>=1.1
matplotlib>=3.5
seaborn>=0.11
plotly>=5.0
joblib>=1.1
## 🧾 License
This project is licensed for educational and non-commercial research use only.



---
### 🔍 Pollution Level Visualization

![Pollution Graph](images/air_pollution_chart.png)

### 📉 Prediction Result Page

![Prediction Result](images/prediction_result.png)




