import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder


def feature_extract():

    #Local Test section
    # data_path = './data/'
    # train_df = np.load(data_path + 'clean_data_train.pkl', allow_pickle=True)
    # test_df = np.load(data_path + 'clean_data_test.pkl', allow_pickle=True)

    s3 = S3FileSystem()
    # S3 bucket directory (data warehouse)
    DIR_wh = 's3://ece5984-s3-pgoda/Project/transformed'                                  # Insert here
    # Get data from S3 bucket as a pickle file
    train_df = np.load(s3.open('{}/{}'.format(DIR_wh, 'clean_data_train.pkl')), allow_pickle=True)
    test_df = np.load(s3.open('{}/{}'.format(DIR_wh, 'clean_data_test.pkl')), allow_pickle=True)


    # Create new dataframe for target Variable or label column for supervised learning
    target_train = pd.DataFrame(train_df['is_fraud'])
    target_test = pd.DataFrame(test_df['is_fraud'])
    #print(target_train)
    #print(target_test)

    # Selecting the Features
    features_scalers = ['cc_num', 'amt', 'unix_time']
    features_labels = ['merchant', 'category', 'gender']
    features = features_scalers + features_labels

    # feature preprossing on the dataset
    # Scaling the dataframe
    scaler = MinMaxScaler()
    train_df[features_scalers] = scaler.fit_transform(train_df[features_scalers])
    test_df[features_scalers] = scaler.fit_transform(test_df[features_scalers])

    # Initialize the LabelEncoder
    encoder = LabelEncoder()
    # Encode the text columns for the train and test datasets
    train_df[features_labels] = train_df[features_labels].apply(encoder.fit_transform)
    test_df[features_labels] = test_df[features_labels].apply(encoder.fit_transform)

    feature_transform_train = pd.DataFrame(columns=features, data=train_df, index=train_df.index)
    feature_transform_test = pd.DataFrame(columns=features, data=test_df, index=test_df.index)

    # pd.set_option('display.max_columns', None)
    # print(feature_transform_train)
    # print(feature_transform_test)


    # Push extracted features to data warehouse
    DIR_aapl = 's3://ece5984-s3-pgoda/Project/features'                               # Insert here
    with s3.open('{}/{}'.format(DIR_aapl, 'x_train.pkl'), 'wb') as f:
        f.write(pickle.dumps(feature_transform_train))
    with s3.open('{}/{}'.format(DIR_aapl, 'x_test.pkl'), 'wb') as f:
        f.write(pickle.dumps(feature_transform_test))
    with s3.open('{}/{}'.format(DIR_aapl, 'y_train.pkl'), 'wb') as f:
        f.write(pickle.dumps(target_train))
    with s3.open('{}/{}'.format(DIR_aapl, 'y_test.pkl'), 'wb') as f:
        f.write(pickle.dumps(target_test))

#Local Test section
#feature_extract()