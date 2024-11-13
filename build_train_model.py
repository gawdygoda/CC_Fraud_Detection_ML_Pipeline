import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd
import tempfile
from xgboost import XGBClassifier

# # Local Test section
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
# import seaborn as sns

def build_train():

    # # Local Test section
    # data_path = './data/'
    # X_train = np.load(data_path + 'x_train.pkl', allow_pickle=True)
    # X_test = np.load(data_path + 'x_test.pkl', allow_pickle=True)
    # y_train = np.load(data_path + 'y_train.pkl', allow_pickle=True)
    # y_test = np.load(data_path + 'y_test.pkl', allow_pickle=True)
    # encoder_dict = np.load(data_path + 'encoder.pkl', allow_pickle=True)

    s3 = S3FileSystem()
    # S3 bucket directory (data warehouse)
    DIR_train = 's3://ece5984-s3-pgoda/Project/features'                       # Insert here
    DIR_test = 's3://ece5984-s3-pgoda/Project/features'                       # Insert here
    DIR_model = 's3://ece5984-s3-pgoda/Project/model'                           # Insert here
    DIR_prediction = 's3://ece5984-s3-pgoda/Project/prediction'                     # Insert here


    X_train = np.load(s3.open('{}/{}'.format(DIR_train, 'x_train.pkl')), allow_pickle=True)
    X_test = np.load(s3.open('{}/{}'.format(DIR_train, 'x_test.pkl')), allow_pickle=True)
    y_train = np.load(s3.open('{}/{}'.format(DIR_test, 'y_train.pkl')), allow_pickle=True)
    y_test = np.load(s3.open('{}/{}'.format(DIR_test, 'y_test.pkl')), allow_pickle=True)

    #Load the encoders we created when we built the model
    encoder_dict = np.load(s3.open('{}/{}'.format(DIR_model, 'encoder.pkl')), allow_pickle=True)

    model = XGBClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    # # Local Test section
    # accuracy = accuracy_score(y_test, prediction)
    # confusionMatrix = confusion_matrix(y_test, prediction)
    #
    # print("Accuracy:", accuracy)
    # auc_roc = roc_auc_score(y_test, prediction)
    # print("AUC-ROC:", auc_roc)
    #
    # # Plotting confusion matrix
    # plt.figure(figsize=(5, 5))
    # sns.heatmap(confusionMatrix, annot=True, annot_kws={"fontsize": 12}, fmt="d", cmap="Blues", cbar=False, linewidths=0.5, linecolor="black")
    # plt.title("Credit Card Fraud Confusion Matrix", fontsize=14)
    # plt.xlabel("Predicted", fontsize=14)
    # plt.ylabel("Actual", fontsize=14)
    # plt.xticks(ticks=[0.5, 1.5], labels=["Not Fraud", "Fraud"], fontsize=10)
    # plt.yticks(ticks=[0.5, 1.5], labels=["Not Fraud", "Fraud"], fontsize=10)
    # plt.show()

    #Craete new dataframe with predictions
    X_test_WithPredictions_df = pd.DataFrame(X_test)
    X_test_WithPredictions_df['predictions'] = prediction

    # # Local Test section
    # print(encoder_dict.classes_)

    #Decode the feature label columns
    features_labels = ['merchant', 'category', 'gender']
    for feature in features_labels:
        encoder = encoder_dict[feature]
        X_test_WithPredictions_df[feature] = encoder.inverse_transform(X_test_WithPredictions_df[feature])

    # Local Test section
    # print(X_test_WithPredictions_df.shape)
    # print(X_test_WithPredictions_df)
    # Save model
    # model.save_model(f"{data_path}model.json")

    # Save model temporarily
    with tempfile.TemporaryDirectory() as tempdir:
        model.save_model(f"{tempdir}/model.json")
        # Push saved model to S3
        s3.put(f"{tempdir}/model.json", f"{DIR_model}/model.json")

    # Push prediction data to S3 bucket warehouse
    with s3.open('{}/{}'.format(DIR_prediction, 'X_test_WithPrediction.pkl'), 'wb') as f:
        f.write(pickle.dumps(X_test_WithPredictions_df))

# # Local Test section
# build_train()