import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import os
import pickle



def ingest_data():

    data_path = './data/'
    train_file_name = 'fraudTrain.csv'
    test_file_name = 'fraudTest.csv'

    full_path_trainData = os.path.join(data_path, train_file_name)
    full_path_testData = os.path.join(data_path, test_file_name)

    # Load the CSV files
    data_train = pd.read_csv(full_path_trainData)
    data_test = pd.read_csv(full_path_testData)

    #Push data from test and training set to S3 data lake as pickle file
    s3 = S3FileSystem()
    # S3 bucket directory
    DIR = 's3://ece5984-s3-pgoda/Project/batch_ingest'                        # Insert here
    # Push data to S3 bucket as a pickle file
    with s3.open('{}/{}'.format(DIR, 'dataTrain.pkl'), 'wb') as f:
        f.write(pickle.dumps(data_train))
    with s3.open('{}/{}'.format(DIR, 'dataTest.pkl'), 'wb') as f:
        f.write(pickle.dumps(data_test))

