import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor, CatBoostClassifier, CatBoost
from math import sin, cos, sqrt, atan2, radians


# # read data in pandas dataframe
df_train =  pd.read_csv('C:/Users/vinni/OneDrive/Desktop/new-york-city-taxi-fare-prediction/train.csv', nrows = 1_000_00, parse_dates=["pickup_datetime"])
df_test = pd.read_csv('C:/Users/vinni/OneDrive/Desktop/new-york-city-taxi-fare-prediction/test.csv', parse_dates=["pickup_datetime"])


#Remove rows with bad latitude and longitude values
df_train = df_train[(df_train["pickup_latitude"] >39.8) & (df_train["pickup_latitude"] < 41.3)]
df_train = df_train[(df_train["pickup_longitude"] > -75) & (df_train["pickup_longitude"] < -71.8)]
df_train = df_train[(df_train["dropoff_latitude"] >39.8) & (df_train["dropoff_latitude"] < 41.3)]
df_train = df_train[(df_train["dropoff_longitude"] > -75) & (df_train["dropoff_longitude"] < -71.8)]

df_train["pickup_latitude"] = df_train["pickup_latitude"].round(4)
df_train["pickup_longitude"] = df_train["pickup_longitude"].round(4)
df_train["dropoff_latitude"] = df_train["dropoff_latitude"].round(4)
df_train["dropoff_longitude"] = df_train["dropoff_longitude"].round(4)

df_test["pickup_latitude"] = df_test["pickup_latitude"].round(4)
df_test["pickup_longitude"] = df_test["pickup_longitude"].round(4)
df_test["dropoff_latitude"] = df_test["dropoff_latitude"].round(4)
df_test["dropoff_longitude"] = df_test["dropoff_longitude"].round(4)



#Removing 195532 rows with passenger count more than 6 and less than 1

df_train = df_train[(df_train["passenger_count"] <= 6 ) & (df_train["passenger_count"] >= 1)]
# 1.73991458e+02 1.40981500e+02 2.53447675e-02

# removing missing values
df_train = df_train.dropna(how = 'any', axis = 'rows')



def distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
   
    angle = 0.017453292519943295 #math.pi / 180
    x = 0.5 - np.cos((lat2 - lat1) * angle) / 2 + np.cos(lat1 * angle) * np.cos(lat2 * angle) * (1 - np.cos((lon2 - lon1) * angle)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(x))

def add_travel_vector_features(df):
    # print(df.dtypes)
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['weekday'] = df['pickup_datetime'].dt.weekday
    df['date'] = df['pickup_datetime'].dt.day
    # df['dayweek'] = df['pickup_datetime'].dt.dayweek
    df['hour'] = df['pickup_datetime'].dt.hour
    df['minute'] = df['pickup_datetime'].dt.minute


    
    df['distance_euclidean'] = distance(df['pickup_latitude'], df['pickup_longitude'], \
                                         df['dropoff_latitude'], df['dropoff_longitude'])

    # feature extraction: distance to specific location
    nyc = (40.7128, -74.0060)
    jfk = (40.6413, -73.7781)
    ewr = (40.6895, -74.1745)
    Manhattan = (40.758896, 73.985130)
    
    df['distance_pickup_to_nyc'] = distance(df['pickup_latitude'], df['pickup_longitude'], nyc[0], nyc[1])
    df['distance_pickup_to_jfk'] = distance(df['pickup_latitude'], df['pickup_longitude'], jfk[0], jfk[1])
    df['distance_pickup_to_ewr'] = distance(df['pickup_latitude'], df['pickup_longitude'], ewr[0], ewr[1])
    df['distance_pickup_mnh'] = distance(df['pickup_latitude'], df['pickup_longitude'], Manhattan[0], Manhattan[1])
    df['distance_dropoff_to_nyc'] = distance(df['dropoff_latitude'], df['dropoff_longitude'], nyc[0], nyc[1])
    df['distance_dropoff_to_jfk'] = distance(df['dropoff_latitude'], df['dropoff_longitude'], jfk[0], jfk[1])
    df['distance_dropoff_to_ewr'] = distance(df['dropoff_latitude'], df['dropoff_longitude'], ewr[0], ewr[1])
    df['distance_dropoff_to_mnh'] = distance(df['dropoff_latitude'], df['dropoff_longitude'], Manhattan[0], Manhattan[1])

add_travel_vector_features(df_train)

df_train = df_train[(df_train['year'] >=1990) & (df_train['year'] <= 2018)]
df_train = df_train[(df_train['month'] >=1) & (df_train['month'] <= 12)]
df_train = df_train[(df_train['hour'] >=0) & (df_train['hour'] <= 24)]
df_train = df_train[(df_train['minute'] >=0) & (df_train['minute'] <= 60)]

df_train = df_train[(df_train['distance_euclidean'] > 0.3) & (df_train['distance_euclidean'] < 101)]
df_train = df_train[(df_train['fare_amount'] > 2.5) & (df_train['fare_amount'] < 130)]

add_travel_vector_features(df_test)



train_X = np.array(df_train.drop(columns=['pickup_datetime', 'key', 'passenger_count', 'fare_amount']))
test_X = np.array(df_test.drop(columns=['pickup_datetime', 'passenger_count', 'key']))


train_y = np.array(df_train['fare_amount'])

clf = RandomForestRegressor(n_jobs=-1)

# clf = RandomForestClassifier()
clf.fit(train_X, train_y)
test_y = clf.predict(test_X)


print(test_y)


submission = pd.DataFrame(
    {'key': df_test.key, 'fare_amount': test_y},
    columns = ['key', 'fare_amount'])
submission.to_csv('C:/Users/vinni/OneDrive/Desktop/new-york-city-taxi-fare-prediction/submission.csv', index = False)

