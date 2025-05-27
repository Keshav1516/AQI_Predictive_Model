## Important Libraries
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

## Load the json data
with open("C:\\Users\\keshav\\OneDrive\\Desktop\\AllStationRealtimeData.json") as f:
    data= json.load(f)
    
data['Locations'][5]

sensor_name= set()
for station in data['Locations']:
    for sensor in station['airquality']:
        sensor_name.add(sensor['sensorName'].lower())

print(f"Available Sensor Name in DataSet: \n", sorted(sensor_name))

# Converting Json into Data Frame.
pollutants = ['pm25', 'pm10', 't', 'h', 'so2', 'no2', 'o3']
records=[]
for station in data['Locations']:
    record= { 'lat': float(station['lat']),
              'lon': float(station['lon'])}
    
    # Initialze we giving polluant assigning with None.
    for pollutant in pollutants:
        record[pollutant]=None
        
    # Fill the value with existing one.
    for sensor in station['airquality']:
        name=sensor['sensorName'].lower()
        if name in pollutants:
            record[name]= sensor['sensorData']
    records.append(record)

df=pd.DataFrame(records)

print("\n All Staion Data:\n", df.head())

df.head()

# Converting to mumeric and drop rows with too many missing values
for col in pollutants:
    df[col]= pd.to_numeric(df[col],errors='coerce')

# Showing how much the missing values are there in Dataset.
print("\n Missing value count:")
print(df[pollutants].isnull().sum())

# Droping the rows with the too many missing values.
df= df.dropna(subset=['pm25', 'pm10', 't', 'h', 'so2', 'no2', 'o3'])

print(f" Remaining rows after dropping: {len(df)}")

# Checkimg the outliers in dataset.
for col in pollutants:
    plt.figure(figsize=(8,6))
    sns.boxplot(x=df[col], color='orange')
    plt.title(f"Boxplot for {col.upper()}")
    plt.xlabel(col)
    plt.show()
    
## We have some outliers in pollutant column like pm25, pm10, t, h, no3, so2, o3. 
# Creating function to don’t want to lose rows, So I tried cap values at the IQR bounds.
def cap_outlier_iqr(df,column):
    Q1= df[column].quantile(0.25)
    Q3= df[column].quantile(0.75)
    IQR= Q1-Q3
    lower= Q1-1.5*IQR  
    upper=Q3-1.5*IQR
    df[column]=np.where(df[column]<lower, lower,df[column])
    df[column]=np.where(df[column]>upper, upper,df[column])
    return df

for col in pollutants:
    if df[col].isnull().all():
        continue
    df= cap_outlier_iqr(df, col)

# Recheckimg the outliers in dataset.
for col in pollutants:
    plt.figure(figsize=(8,6))
    sns.boxplot(x=df[col], color='orange')
    plt.title(f"Boxplot for {col.upper()}")
    plt.xlabel(col)
    plt.show()

## Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import NotFittedError

# Training model
x= df[['lat','lon']]
trained_model= {}
models= { 'RandomForest': RandomForestRegressor(),
         'KNeighbors' : KNeighborsRegressor(),
         'GradientBoosting': GradientBoostingRegressor(),
         'XGB': XGBRegressor()}
print("\n Training Models:")
for target in pollutants:
    if df[target].isnull().all():
        print(f" Skipping {target}: no-data available.")
        continue
    y=df[target].dropna()
    x_target= x.loc[y.index] # match x with available y
    x_train, x_test, y_train, y_test= train_test_split(x_target, y, test_size=0.2, random_state=42)

    trained_model[target] = {}
    for name, model in models.items():
        try:
            model.fit(x_train, y_train)
            y_pred= model.predict(x_test)
            rmse= mean_squared_error(y_test, y_pred, squared=False)
            print(f"{target.upper()} | {name} RMSE: {rmse:.2f}")
            trained_model[target][name]= model
        except Exception as e:
            print(f"Error training {target} with {name}: {e}")

## As see in the output that RMSE (Root Mean Squares Error) square root of the average of the squared differences 
## between the predicted values and the actual values.
 
## Prediction    
# Creating the prediction function. 
def predict_pollutants(lat, lon):
    input_data = np.array([[lat, lon]])
    predictions = {}
    
    for target in trained_model:
        model = trained_model[target].get('RandomForest')  # pick model
        if model:
            try:
                predictions[target] = round(model.predict(input_data)[0], 2)
            except NotFittedError:
                predictions[target] = 'N/A'
        else:
            predictions[target] = 'N/A'

    return predictions
# Prediction at a sample location
lat_input= -29.5
lon_input= -67.5

predicted= predict_pollutants(lat_input, lon_input)

print(f"Prediction at ({lat_input},{lon_input}):")
for key, value in predicted.items():
    print(f"{key.upper()}:{value}")
    
## We are getting error seeing the same predicted value (57.0) for all pollutants, even when you change the latitude and longitude — 
## which means our model is likely overfitting or returning a constant prediction. because of muilti-model uses. So we have to go this 
## one one model according which we check which model with the lowest RMSE for each pollutant and store it for prediction. And predict actual prediction

# I am creating a separate model going head with RandomForestRegressor. 

pollutants = ['pm25', 'pm10', 't', 'h', 'so2', 'no2', 'o3']
pollutants = [col for col in pollutants if col in df.columns]

trained_models = {}

# Train one model per pollutant
for col in pollutants:
    y = df[col]
    X = df[['lat', 'lon']]

    model = RandomForestRegressor()
    model.fit(X, y)

    trained_models[col] = model
    print(f" Trained model for {col.upper()} \n")

# Creating define prediction function
def predict_pollutants(lat, lon):
    input_data = np.array([[lat, lon]])
    predictions = {}
    for col, model in trained_models.items():
        predictions[col.upper()] = round(model.predict(input_data)[0], 2)
    return predictions

# Test with real lat/lon
lat_input= -29.5
lon_input= -67.5

predicted= predict_pollutants(lat_input, lon_input)

print(f"Prediction at ({lat_input},{lon_input}):")
for key, value in predicted.items():
    print(f"{key.upper()}:{value}")