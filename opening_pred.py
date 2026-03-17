import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Retrieve the data for both train and test file
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

#Print top 5 records
print(data_train.head(5))
print(data_test.head(5))

print("-"*75)

#Print info train file
print(data_train.info())

print("-"*75)
#Print statistics train value
print(data_train.describe())

#Lists with columns of both numercial and categorical data
col_num = data_train.select_dtypes(include=['int64','float64']).columns
col_cat = data_train.select_dtypes(include=['object']).columns

#data_train['stock_id'] = data_train['stock_id'].astype('int64')

print("Numerical columns: ", col_num)
print("Categorical columns: ", col_cat)

#Test for empty values in the training data
for col in data_train.columns:
    if data_train[col].isnull().sum() > 0:
        print(col, data_train[col].isnull().sum())

col_headers=[col for col in data_train.columns if col not in ['stock_id','target']]
x=data_train[col_headers]
y=data_train.iloc[:,-1]

#x data for test data
x_test_kaggle=data_test[col_headers].values

#Separate train file in train and test set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

print("-"*75)

#Create the model
model=XGBClassifier(objective='binary:logistic',n_estimators=1000,learning_rate=0.05)
model.fit(x_train, y_train)

#Predict the model
y_pred=model.predict_proba(x_test)[:,1]
print("ROC AUC Score: ", roc_auc_score(y_test, y_pred))

#Predictions for the kaggle test set
y_pred_kaggle=model.predict_proba(x_test_kaggle)[:,1]
#Create dataframe for submission
submission = pd.DataFrame({'id': data_test['id'], 'target': y_pred_kaggle})
                          
#Write the submission dataframe to a CSV file
submission.to_csv('submission.csv', index=False)

#Plot feature importance
plot_importance(model)
plt.show()

#Transform probabilities to binary predictions using a threshold of 0.5
y_pred_binary = (y_pred >= 0.5).astype(int)
#Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:\n", cm) 

#Make a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
