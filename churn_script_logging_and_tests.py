"""
This module tests and executes the functions in churn_library.py
Author: Dmitriy K.
Created: 6/10/20
"""

import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        df = cl.import_data("./data/bank_data.csv")
        perform_eda(df)
        assert 'Churn' in df.columns
        assert list(df['Churn'].unique()) == [0,1]
    except AssertionError as err:
        logging.error("Testing perform_eda: column named 'Churn' is not in the df,\
                      it must be in the dataframe, and must be either 0 or 1 to perform EDA")
        raise err

    try:
        files = ['churn_distribution.png','customer_age_distribution.png','heatmap.png',
                 'marital_status_distribution.png','total_transaction_distribution.png']
        for file in files:
            assert os.path.isfile('images/eda/' + file)
        logging.info("Testing test_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda:\
                      some or all of the following files are not in images/eda dir:%s",file)

def test_classification_report_image():
    '''
    test classification report function
    '''
    try:
        df = cl.import_data("./data/bank_data.csv")
        cl.classification_report_image(df)
        files = ["logistics_results.png","rf_results.png","roc_curve_result.png"]
        for file in files:
            assert os.path.isfile('images/results/' + file)
        logging.info("Testing classification_report_image: SUCCESS")
    except AssertionError as err:
        logging.error("Testing classification_report_image:\
                      some or all of the following files are not in images/results dir:%s",files)

def test_feature_importance_plot(feature_importance_plot):
    '''
    test feature importance plot
    '''
    try:
        assert os.path.isfile('images/results/feature_importance_plot.png')
        logging.info("Testing feature_impotance_plot: SUCCESS")
    except AssertionError as err:
        logging.error("Testing feature_impotance_plot: feature_importance_plot.png\
                      is not in the images/results dir")

def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    category_lst = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
    ]
    try:
        df = cl.import_data("./data/bank_data.csv")
        df = encoder_helper(df, category_lst)
        cols = list(df.columns)
        expected_cols = ['Gender_Churn','Education_Level_Churn','Marital_Status_Churn',
                         'Income_Category_Churn','Card_Category_Churn']
        assert all(i in cols for i in expected_cols)
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper:\
                      missing some or all of these columns are missing:%s",expected_cols)

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        df = cl.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        assert X_train.shape[1] == 19
        assert X_test.shape[1] == 19
        assert list(y_train.unique()) == [0,1]
        assert list(y_test.unique()) == [0,1]
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering:\
                      there must be 19 features and the target column must contain either 0 or 1")

def test_train_models(train_models):
    '''
    test train_models
    '''
    df = cl.import_data("./data/bank_data.csv")
    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)
    try:
        assert os.path.isfile('models/rfc_model.pkl')
        assert os.path.isfile('models/logistic_model.pkl')
        logging.info("Testing test_train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models:\
                      rfc_model.pkl and logistic_model.pkl must be in the models dir")

if __name__ == "__main__":
    test_import(cl.import_data)
    test_eda(cl.perform_eda)
    test_classification_report_image()
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_train_models(cl.train_models)
