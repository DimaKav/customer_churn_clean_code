"""
This module performs EDA, feature engineering, and model training.
Author: Dmitriy K.
Created: 6/10/20
"""

# import libraries
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve, classification_report
sns.set()
os.environ['QT_QPA_PLATFORM']='offscreen'

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    df['Churn'].hist()
    plt.savefig('images/eda/churn_distribution.png')
    df['Customer_Age'].hist()
    plt.savefig('images/eda/customer_age_distribution.png')
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/marital_status_distribution.png')
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig('images/eda/total_transaction_distribution.png')
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('images/eda/heatmap.png')

def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
    output:
            df: pandas dataframe with new columns for
    '''
    df = df.copy()
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    for col in category_lst:
        groups = df.groupby(col).mean()['Churn']
        cat_list = []
        for val in df[col]:
            cat_list.append(groups.loc[val])
        df[col + '_Churn'] = cat_list

    return df

def perform_feature_engineering(df):
    '''
    performs basic feature engineering via the encoder_helper and splits the dataset
    for testing and training
    input:
              df: pandas dataframe
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    category_lst = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
    ]
    df = encoder_helper(df, category_lst)
    y = df['Churn']
    X = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']
    X[keep_cols] = df[keep_cols]
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def get_classification_report_args(df):
    '''
    helper function which outputs data necessary for
    obtaining a classification report via classification_report_image()
    input:
              df: pandas dataframe
    output:
              everything needed for classification_report_image
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=200)

    rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    y_test_preds_rf = rfc.predict(X_test)
    y_test_preds_lr = lrc.predict(X_test)

    return [y_test,y_test_preds_lr,y_test_preds_rf]


def classification_report_image(df):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    '''
    y_test,y_test_preds_lr,y_test_preds_rf = get_classification_report_args(df)

    rf_test_report = classification_report(y_test, y_test_preds_rf,output_dict=True)
    lr_test_report = classification_report(y_test, y_test_preds_lr,output_dict=True)
    reports = [rf_test_report,lr_test_report]
    fnames = ['images/results/rf_results.png','images/results/logistics_results.png']

    for report, fname in zip(reports, fnames):
        temp_df = pd.DataFrame(report).reset_index()
        #define figure and axes
        ax = plt.subplots()[1]
        table = ax.table(cellText=temp_df.values,
                         colLabels=temp_df.columns,
                         loc='center')
        #modify table
        table.set_fontsize(20)
        ax.axis('off')
        #save table
        plt.savefig(fname, dpi=200)

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # instantiate and fit the models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=300)

    rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # plots
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('images/results/roc_curve_result.png')

    # save best model
    joblib.dump(rfc, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

def get_xdata(df):
    '''
    helper for obtaining X data for feature_importance_plot()
    input:
              df: pandas dataframe
    output:
              X: x data
    '''
    category_lst = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
    ]
    df = encoder_helper(df, category_lst)
    X = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']
    X[keep_cols] = df[keep_cols]

    return X

def feature_importance_plot():
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
    output:
             None
    '''
    model = joblib.load('./models/rfc_model.pkl')
    x_data = get_xdata(import_data('data/bank_data.csv'))
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = pd.np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig('images/results/feature_importance.png')

def run_churn_library():
    """runs this entire library of functions"""
    df = import_data('data/bank_data.csv')
    perform_eda(df)
    classification_report_image(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)
    feature_importance_plot()

if __name__ == "__main__":
    run_churn_library()
