import numpy as np
import pandas as pd
import os

# Input filenames and variables
data_path = './data/'
train_file_name = 'fraudTrain.csv'
test_file_name = 'fraudTest.csv'

full_path_trainData = os.path.join(data_path, train_file_name)
full_path_testData = os.path.join(data_path, test_file_name)

# Load the CSV files
data_train = pd.read_csv(full_path_trainData)
data_test = pd.read_csv(full_path_testData)

# After initial EDA, we know the first column name is missing
# so we add it here
data_train.rename(columns={'Unnamed: 0': 'Row Number'}, inplace=True)

# Display the complete data
print("The Training dataset looks like:")
print(data_train)
print(data_train.shape)
print("====================================")

# Set options to show all columns of the dataset
pd.set_option('display.max_columns', None)

# Display all the columns together in the console
print("Display first 5 rows")
print(data_train.head().to_string())
print("====================================")


# Getting a feel of the dataset
# Basic EDA functions

print("Basic Dataframe info")
print(data_train.info())
print("====================================")
print("More detailed Dataframe info")
print(data_train.describe().to_string())
print("====================================")
print("Number of Empty values in each column:")
print(data_train.isnull().sum().sort_values(ascending = False))
print("====================================")
print("Number of Unique values in each column:")
print(data_train.apply(pd.Series.nunique))
print("====================================")
print("Are there duplicate rows?")
print(data_train.duplicated())
print("====================================")


# After EDA, we have the following notes:
# - no null values in any columns
# - 1296675 rows of data
# - 1274791 unique transaction date/times
# - 983 CC numbers
# - 983 unique street addresses
# - 1296675 unique transaction numbers
# - 693 unique merchants

# Number of Unique values in each column:
# Row Number               1296675
# trans_date_trans_time    1274791
# cc_num                       983
# merchant                     693
# category                      14
# amt                        52928
# first                        352
# last                         481
# gender                         2
# street                       983
# city                         894
# state                         51
# zip                          970
# lat                          968
# long                         969
# city_pop                     879
# job                          494
# dob                          968
# trans_num                1296675
# unix_time                1274823
# merch_lat                1247805
# merch_long               1275745
# is_fraud                       2


# Dividing the raw dataset for each company individual company
# raw_data.columns = raw_data.columns.swaplevel(0,1)
# print("Describe Dataframe after swaplevel")
# print(raw_data.describe().to_string())
# print("====================================")
# raw_data.sort_index(axis=1, level=0, inplace=True)
# print("Describe Dataframe after sort")
# print(raw_data.describe().to_string())
# print("====================================")


# # Dropping rows with NaN in specific columns only
# df_movies = raw_data.dropna(subset=['RATING', 'VOTES'])
# print("Describe Dataframe after drop")
# print(df_movies)
# print("====================================")
#
# # Dropping duplicate rows
# df_movies = df_movies.drop_duplicates()
# print("Describe Dataframe after dup removal")
# print(df_movies)
# print("====================================")