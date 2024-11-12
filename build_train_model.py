import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd
import tempfile
from xgboost import XGBClassifier

# Local Test section
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, confusion_matrix
# import seaborn as sns

def build_train():

    # Local Test section
    # data_path = './data/'
    # X_train = np.load(data_path + 'x_train.pkl', allow_pickle=True)
    # X_test = np.load(data_path + 'x_test.pkl', allow_pickle=True)
    # y_train = np.load(data_path + 'y_train.pkl', allow_pickle=True)
    # y_test = np.load(data_path + 'y_test.pkl', allow_pickle=True)

    s3 = S3FileSystem()
    # S3 bucket directory (data warehouse)
    DIR_train = 's3://ece5984-s3-pgoda/Project/features'                       # Insert here
    DIR_test = 's3://ece5984-s3-pgoda/Project/features'                       # Insert here
    DIR_model = 's3://ece5984-s3-pgoda/Project/model'  # Insert here

    X_train = np.load(s3.open('{}/{}'.format(DIR_train, 'x_train.pkl')), allow_pickle=True)
    X_test = np.load(s3.open('{}/{}'.format(DIR_train, 'x_test.pkl')), allow_pickle=True)
    y_train = np.load(s3.open('{}/{}'.format(DIR_test, 'y_train.pkl')), allow_pickle=True)
    y_test = np.load(s3.open('{}/{}'.format(DIR_test, 'y_test.pkl')), allow_pickle=True)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # confusionMatrix = confusion_matrix(y_test, y_pred)
    #
    # print("Accuracy:", accuracy)
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
    #
    # # Save model
    # model.save_model(f"{data_path}model.json")


    # Save model temporarily
    with tempfile.TemporaryDirectory() as tempdir:
        model.save_model(f"{tempdir}/model.json")
        # Push saved model to S3
        s3.put(f"{tempdir}/model.json", f"{DIR_model}/model.json")

# Local Test section
# build_train()