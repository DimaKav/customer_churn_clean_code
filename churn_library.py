# library doc string


# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

import os
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
    plt.savefig('images/eda/churn.png')
    df['Customer_Age'].hist()
    plt.savefig('images/eda/customer_age.png')
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/marital_status.png')
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig('images/eda/total_trans_ct.png')
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
    df=df.sample(500)
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


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    rf_test_report = classification_report(y_test, y_test_preds_rf,output_dict=True)
    rf_train_report = classification_report(y_train, y_train_preds_rf,output_dict=True)
    lr_test_report = classification_report(y_test, y_test_preds_lr,output_dict=True)
    lr_train_report = classification_report(y_train, y_train_preds_lr,output_dict=True)
    reports = [rf_test_report,rf_train_report,lr_test_report,lr_train_report]
    fnames = ['rftest.png','rftrain.png','lrtest.png','lrtrain.png']
    
    for report, fname in zip(reports, fnames):
        temp_df = pd.DataFrame(report).reset_index()
        #define figure and axes
        fig, ax = plt.subplots()
        table = ax.table(cellText=temp_df.values, 
                         colLabels=temp_df.columns, 
                         loc='center')
        #modify table
        table.set_fontsize(20)
        ax.axis('off')
        #save table
        plt.savefig(fname, dpi=200)
        
def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90);
    plt.savefig(output_pth)
    
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
    lrc = LogisticRegression(max_iter=200)

    rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # plots
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('two_models.png')
    
    # save best model
    joblib.dump(rfc, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    

# df = pd.read_csv('data/bank_data.csv')
# X_train, X_test, y_train, y_test = perform_feature_engineering(df)
# train_models(X_train, X_test, y_train, y_test)