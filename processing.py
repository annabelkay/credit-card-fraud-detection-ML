# Pre-processing the credit card fraud dataset from Kaggle.
# Here we experiment with two class balancing techniques - SMOTE and Near Miss - before building our machine learning models.

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report 
from imblearn.over_sampling import SMOTE 

# Loading the dataset
data = pd.read_csv('creditcard.csv')

# Understanding the original shape of the data - number of rows and columns.
data.shape

# Investigating whether there is multicollinearity in the dataset - that requires cleaning. 
corr = data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

# Seeing whether there are any empty cells of data.
data.isnull().values.any()





# Assessing the balance of the class label (fraud vs no fraud).
data['Class'].value_counts().plot.bar()
print('Proportion of the classes in the data:')
print(data['Class'].value_counts() / len(data))

# Organising the class data.
X = np.array(data.loc[:, data.columns != 'Class'])
y = np.array(data.loc[:, data.columns == 'Class']).reshape(-1, 1)

# Standardising the data.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the data into training and testing datasets using stratified sampling.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 4, shuffle = True, stratify = y)




# Performing without SMOTE or Near Miss

# Importing logistic regression model and accuracy_score metric.
clf = LogisticRegression(solver = 'lbfgs')

# Loading the Logistic Regression object.
lr = LogisticRegression() 
  
# Training the model on the training set.
lr.fit(X_train, y_train.ravel()) 
predictions = lr.predict(X_test) 
  
# Printing the classification report for evaluation.
print(classification_report(y_test, predictions)) 





# Perfoming with Near Miss

# Data before undersampling - label counts.
print("Before Undersampling, label counts of '1': {}".format(sum(y_train == 1))) 
print("Before Undersampling, label counts of '0': {} \n".format(sum(y_train == 0))) 
  
# Applying Near Miss to the training data sets.
from imblearn.under_sampling import NearMiss 
nr = NearMiss() 
X_train_miss, y_train_miss = nr.fit_sample(X_train, y_train.ravel()) 
  
# Data after undersampling - shape of X and y training data.
print('After Undersampling, the shape of X_train: {}'.format(X_train_miss.shape)) 
print('After Undersampling, the shape of y_train: {} \n'.format(y_train_miss.shape)) 
  
# Data after undersampling - label counts.
print("After Undersampling, label counts of '1': {}".format(sum(y_train_miss == 1))) 
print("After Undersampling, label counts of '0': {}".format(sum(y_train_miss == 0)))

# Training the model on train set.
lr2 = LogisticRegression() 
lr2.fit(X_train_miss, y_train_miss.ravel()) 
predictions = lr2.predict(X_test) 
  
# Printing the classification report for evaluation.
print(classification_report(y_test, predictions)) 

# We need to balance the test set - as there are still over 80,000 samples in the 'no fraud' label.
# The imbalanced nature of this data set is extreme, thus this step is imperative to our research.
print("Before Undersampling, counts of label '1': {}".format(sum(y_test == 1))) 
print("Before Undersampling, counts of label '0': {} \n".format(sum(y_test == 0))) 
  
# Applying Near Miss to our test sets. 
from imblearn.under_sampling import NearMiss 
nr = NearMiss() 
X_test_miss, y_test_miss = nr.fit_sample(X_test, y_test.ravel()) 

# Data after undersampling - shape of training data.  
print('After Undersampling, the shape of train_X: {}'.format(X_test_miss.shape)) 
print('After Undersampling, the shape of train_y: {} \n'.format(y_test_miss.shape)) 
  
# Data after undersampling - count of label data.
print("After Undersampling, counts of label '1': {}".format(sum(y_test_miss == 1))) 
print("After Undersampling, counts of label '0': {}".format(sum(y_test_miss == 0)))




# Performing with SMOTE

# Data before oversampling - count of label data.
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 

# Applying SMOTE to our training sets
sm = SMOTE(random_state = 2) 
X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel()) 

# Data after oversampling - shape of training data.
print('After OverSampling, the shape of train_X: {}'.format(X_train_new.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_new.shape)) 
  
# Data after oversampling - count of label data.
print("After OverSampling, counts of label '1': {}".format(sum(y_train_new == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_new == 0))) 

# Training the model on train set.
lr1 = LogisticRegression() 
lr1.fit(X_train_new, y_train_new.ravel()) 
predictions = lr1.predict(X_test) 
  
# Printing classification report for evaluation.
print(classification_report(y_test, predictions)) 



# Now we create our final data set - ready for our algorithm implementations in MATLAB.
# At the start of this process, we were using data from our SMOTE transformation, due to it's ability to produce strong accuracy and recall results. 
# However, after using SVM and MLP in MATLAB, we realised that oversampling our data gave us accuracy results that were too high and over optimistic (99%+). 
# This could have been because the classes were so extremely imbalanced that the minority class created too many synthetic results after SMOTE that were too similar to the original set. 
# This was not authentic. We also found that the data set was too large for the scale of project we were undertaking, slowing our algorithms down and preventing us from making the most out of our optimisation techniques.
# Therefore, we decide on using Near Miss. Additionally, we add white gaussian noise to our data set to improve generalisation performance.
# Let's check the size of each set before we concatenate our tables.

# Checking the size of the X train data 
len(X_train_miss)

# Checking the size of the y train data 
len(y_train_miss)

# Checking the size of the X test data 
len(y_test_miss)

# Checking the size of the y test data 
len(X_test_miss)

# Combining the training sets together from Near Miss
dfTrain = pd.concat([pd.DataFrame(X_train_miss), pd.DataFrame(y_train_miss)], axis=1)

# Combing the test sets together from Near Miss
dfTest = pd.concat([pd.DataFrame(X_test_miss), pd.DataFrame(y_test_miss)], axis=1)

# Naming our variables - the variable names were lost in the process of concatenation. 
dfTest= dfTest.set_axis(['Time','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class'], axis=1, inplace=False)
dfTrain= dfTrain.set_axis(['Time','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class'], axis=1, inplace=False)

# Combining the training and test sets together to form our final data set.
finalData = pd.concat([pd.DataFrame(dfTest), pd.DataFrame(dfTrain)], axis=0)
finalData.to_csv('clean_data.csv')

This marks the end of the data processing stage of our analysis. We will now load the data into MATLAB and build our SVM and MLP models.
