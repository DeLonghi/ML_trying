import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model

# # read data in pandas dataframe
df_train =  pd.read_csv('C:/Users/vinni/OneDrive/Desktop/new-york-city-taxi-fare-prediction/train.csv', nrows = 1_000_00, parse_dates=["pickup_datetime"])
df_test = pd.read_csv('C:/Users/vinni/OneDrive/Desktop/new-york-city-taxi-fare-prediction/test.csv')


# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(df_train)
add_travel_vector_features(df_test)


#Remove rows with bad latitude and longitude values
df_train = df_train[(df_train["pickup_latitude"] >39.8) & (df_train["pickup_latitude"] < 41.3)]
df_train = df_train[(df_train["pickup_longitude"] > -75) & (df_train["pickup_longitude"] < -71.8)]
df_train = df_train[(df_train["dropoff_latitude"] >39.8) & (df_train["dropoff_latitude"] < 41.3)]
df_train = df_train[(df_train["dropoff_longitude"] > -75) & (df_train["dropoff_longitude"] < -71.8)]

#Removing 195532 rows with passenger count more than 6 and less than 1

df_train = df_train[(df_train["passenger_count"] <= 6 ) & (df_train["passenger_count"] >= 1)]
# 1.73991458e+02 1.40981500e+02 2.53447675e-02

# list first few rows (datapoints)
print(df_train)

# removing missing values
df_train = df_train.dropna(how = 'any', axis = 'rows')

def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, df.passenger_count))

train_X = get_input_matrix(df_train)
test_X = get_input_matrix(df_test)

train_y = np.array(df_train['fare_amount'])

print(train_X.shape)
print(train_y.shape)


regr = linear_model.LinearRegression()

# train the model using the training sets
regr.fit(train_X, train_y)

print(regr.coef_)

# make predictions using the testing set
test_y = regr.predict(test_X)

submission = pd.DataFrame(
    {'key': df_test.key, 'fare_amount': test_y},
    columns = ['key', 'fare_amount'])
submission.to_csv('C:/Users/vinni/OneDrive/Desktop/new-york-city-taxi-fare-prediction/submission.csv', index = False)

