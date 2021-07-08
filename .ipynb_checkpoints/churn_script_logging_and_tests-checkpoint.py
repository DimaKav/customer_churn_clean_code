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
        logging.error("Testing test_eda: column named 'Churn' is not in the df, it must be in the dataframe, and must be either 0 or 1 to perform EDA")
        raise err
    
    try:
        assert os.path.isfile('images/eda/churn.png')
        assert os.path.isfile('images/eda/customer_age.png')
        assert os.path.isfile('images/eda/marital_status.png')
        assert os.path.isfile('images/eda/total_trans_ct.png')
        assert os.path.isfile('images/eda/heatmap.png')
        logging.info("Testing test_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing test_eda: churn.png, customer_age.png, marital_status.png, total_trans_ct.png, heatmap.png must all be in the images/eda directory")
        

def test_classification_report_image(classification_report_image):
    '''
    test classification report function
    '''
    
def test_feature_importance_plot(feature_impotance_plot):
    '''
    test feature importance plot
    '''
        
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
        logging.info("Testing test_encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(f"Testing encoder_helper: missing some or all of these columns are missing:{expected_cols}")


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    try:
        df = cl.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        assert X_train.shape[1] == 19
        assert X_test.shape[1] == 19
        assert list(y_train.unique()) == [0,1]
        assert list(y_test.unique()) == [0,1]
        logging.info("Testing test_perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing test_perform_feature_engineering: there must be 19 features and the target column must contain either 0 or 1")
        

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
        logging.error("Testing train_models: rfc_model.pkl and logistic_model.pkl must be in the models dir")


if __name__ == "__main__":
    test_train_models(cl.train_models)
