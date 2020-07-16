# Credit Card Fraud Detection - Machine Learning (MATLAB and Python)

## Description 🖋️

This repository demonstrates the usage of a Support Vector Machine and a Multi-Layer Perceptron Model to detect credit card fraud.

### Original Dataset:
You can find the original dataset [here](https://www.kaggle.com/mlg-ulb/creditcardfraud).



### Implementation Details:

Here are the relevant implementation details including instructions on how to train and run the models to reproduce the optimised results. 

To test the two models, please run the following MATLAB files:

-	`SVM_Model_Test.m` for the Support Vector Machine model.

-	`MLP_Model_Test.m` for the Multi-Layer Perceptron model. This model needs to be ran twice in order to retrieve the same test results.

You will also find two files containing the Bayesian Hyperparameter Optimisation methods undertaken for both models. These files also include our functions, cross-validation procedures and loss calculations:

-	`SVM_Model_Optimisation.m` for the SVM model.
-	`MLP_Model_Optimisation.m` for the MLP model.

**Clean dataset**:

-	`clean_data.csv` is used for both models (with feature selection/under sampling implementations made in `processing.py`).

We include a python file containing the pre-processing stage of the analysis, where we explore the data. The file also includes an assessment of two data balancing techniques: SMOTE and Near Miss. 

-	`processing.py` for the pre-processing file.
