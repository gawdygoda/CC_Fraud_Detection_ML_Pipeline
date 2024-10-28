import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle

def transform_data():

    s3 = S3FileSystem()
    # S3 bucket directory (data lake)
    DIR = 's3://ece5984-s3-pgoda/Project/batch_ingest' #Pipeline location for DataLake (Remove trailing slash)
    # Get data from S3 bucket as a pickle file
    raw_data_train = np.load(s3.open('{}/{}'.format(DIR, 'dataTrain.pkl')), allow_pickle=True)
    raw_data_test = np.load(s3.open('{}/{}'.format(DIR, 'dataTest.pkl')), allow_pickle=True)

    # After initial EDA, we know the first column name is missing
    # so we add it here
    raw_data_train.rename(columns={'Unnamed: 0': 'Row Number'}, inplace=True)
    raw_data_test.rename(columns={'Unnamed: 0': 'Row Number'}, inplace=True)

    df_train = raw_data_train
    df_test = raw_data_test

    # Push cleaned data to S3 bucket warehouse
    DIR_wh = 's3://ece5984-s3-pgoda/Project/transformed'                     # Insert here
    with s3.open('{}/{}'.format(DIR_wh, 'clean_data_train.pkl'), 'wb') as f:
        f.write(pickle.dumps(df_train))
    with s3.open('{}/{}'.format(DIR_wh, 'clean_data_test.pkl'), 'wb') as f:
        f.write(pickle.dumps(df_test))




