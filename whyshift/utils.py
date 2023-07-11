import argparse
import pandas as pd 
import numpy as np 
import pandas as pd 
import json
from datetime import datetime
import re
import os
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize

location_list = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",  'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal']

def preprocess(dir):
    try:
        data = pd.read_csv(dir)
    except:
        raise FileNotFoundError('File does not exist: {}'.format(dir))
    X = data
    X["Start_Time"] = pd.to_datetime(X["Start_Time"])
    # Extract year, month, weekday and day
    X["Year"] = X["Start_Time"].dt.year
    X["Month"] = X["Start_Time"].dt.month
    X["Weekday"] = X["Start_Time"].dt.weekday
    X["Day"] = X["Start_Time"].dt.day
    # Extract hour and minute
    X["Hour"] = X["Start_Time"].dt.hour
    X["Minute"] = X["Start_Time"].dt.minute
    features_to_drop = ["ID", "Start_Time", "End_Time", "End_Lat", "End_Lng", "Description", "Number", "Street", "County", "Zipcode", "City", "Country", "Timezone", "Airport_Code", "Weather_Timestamp", "Wind_Chill(F)", "Turning_Loop", "Sunrise_Sunset", "Nautical_Twilight", "Astronomical_Twilight"]
    X = X.drop(features_to_drop, axis=1)
    X.drop_duplicates(inplace=True)
    X = X[X["Side"] != " "]
    X = X[X["Pressure(in)"] != 0]
    X = X[X["Visibility(mi)"] != 0]
    X.loc[X["Weather_Condition"].str.contains("Thunder|T-Storm", na=False), "Weather_Condition"] = "Thunderstorm"
    X.loc[X["Weather_Condition"].str.contains("Snow|Sleet|Wintry", na=False), "Weather_Condition"] = "Snow"
    X.loc[X["Weather_Condition"].str.contains("Rain|Drizzle|Shower", na=False), "Weather_Condition"] = "Rain"
    X.loc[X["Weather_Condition"].str.contains("Wind|Squalls", na=False), "Weather_Condition"] = "Windy"
    X.loc[X["Weather_Condition"].str.contains("Hail|Pellets", na=False), "Weather_Condition"] = "Hail"
    X.loc[X["Weather_Condition"].str.contains("Fair", na=False), "Weather_Condition"] = "Clear"
    X.loc[X["Weather_Condition"].str.contains("Cloud|Overcast", na=False), "Weather_Condition"] = "Cloudy"
    X.loc[X["Weather_Condition"].str.contains("Mist|Haze|Fog", na=False), "Weather_Condition"] = "Fog"
    X.loc[X["Weather_Condition"].str.contains("Sand|Dust", na=False), "Weather_Condition"] = "Sand"
    X.loc[X["Weather_Condition"].str.contains("Smoke|Volcanic Ash", na=False), "Weather_Condition"] = "Smoke"
    X.loc[X["Weather_Condition"].str.contains("N/A Precipitation", na=False), "Weather_Condition"] = np.nan
    X.loc[X["Wind_Direction"] == "CALM", "Wind_Direction"] = "Calm"
    X.loc[X["Wind_Direction"] == "VAR", "Wind_Direction"] = "Variable"
    X.loc[X["Wind_Direction"] == "East", "Wind_Direction"] = "E"
    X.loc[X["Wind_Direction"] == "North", "Wind_Direction"] = "N"
    X.loc[X["Wind_Direction"] == "South", "Wind_Direction"] = "S"
    X.loc[X["Wind_Direction"] == "West", "Wind_Direction"] = "W"
    features_to_fill = ["Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
    X[features_to_fill] = X[features_to_fill].fillna(X[features_to_fill].mean())
    X.dropna(inplace=True)
    X["Wind_Direction"] = X["Wind_Direction"].map(lambda x : x if len(x) != 3 else x[1:], na_action="ignore")
    X = X[X["Severity"] != 1]
    X = X[X["Severity"] != 4]
    size = len(X[X["Severity"]==3].index)
    df = pd.DataFrame()
    for i in [2,3]:
        S = X[X["Severity"]==i]
        df = df.append(S.sample(size, random_state=42))
    X = df
    X["Severity"] -= 2
    
    scaler = MinMaxScaler()
    features = ['Temperature(F)','Distance(mi)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Start_Lng','Start_Lat','Year', 'Month','Weekday','Day','Hour','Minute']
    X[features] = scaler.fit_transform(X[features])
    categorical_features = set(["Side", "Wind_Direction", "Weather_Condition", "Civil_Twilight"])
    
    for cat in categorical_features:
        X[cat] = X[cat].astype("category")
    X = X.replace([True, False], [1, 0])
    onehot_cols = categorical_features 
    X = pd.get_dummies(X, columns=onehot_cols, drop_first=True)
    return X 
  
    
def get_USACCdata(dir, state):
    X = preprocess(dir)
    X = X[X["State"]==state]
    sample = X
    X.to_csv("./datasets/USAccident/%s.csv"%state)
    y_sample = sample["Severity"]
    X_sample = sample.drop(["Severity", "State", "Start_Lat", "Start_Lng"], axis=1).values
    return X_sample, y_sample