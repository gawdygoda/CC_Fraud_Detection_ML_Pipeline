import numpy as np
import pandas as pd
import pickle


# Raw data
raw_data = np.load(r'data_hw1_1.pkl', allow_pickle=True)   # Load pickle data here
# Display the complete data
print("The Dataset looks like:")
print(raw_data)
print(raw_data.shape)
print("====================================")

# Set options to show all columns of the dataset
pd.set_option('display.max_columns', None)
# Display all the columns together in the console
print("Display first 5 rows")
print(raw_data.head().to_string())
print("====================================")


# Getting a feel of the dataset
# Basic EDA functions

print("Basic Dataframe info")
print(raw_data.info())
print("====================================")
print("More detailed Dataframe info")
print(raw_data.describe().to_string())
print("====================================")
print("Number of Empty values in each column:")
print(raw_data.isnull().sum().sort_values(ascending = False))
print("====================================")
print("Number of Unique values in each column:")
print(raw_data.apply(pd.Series.nunique))
print("====================================")
print("Are there duplicate rows?")
print(raw_data.duplicated())
print("====================================")


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