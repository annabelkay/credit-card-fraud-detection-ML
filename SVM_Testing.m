%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%  Support Vector Machine Model  %%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% See 'Optimisation_MLP_Model.m' to see how we optimised our model and for
% more technical details.

%% Importing the data.

% Clearing previous command history.
clear; close all; clc

% Ensuring randomness of model remains constant.
% This means we are able to obtain the optimal results.
rng('default')

% Loading the data.
data = readtable('clean_data.csv');

% Removing unwanted column.
data.Var1 = [];

%% Splitting the data set.

% Splitting the data using randperm to reduce bias.
split_size = 0.15;
N = size(data,1);
split = false(N,1);
split(1:round(split_size*N)) = true;
split = split(randperm(N));

% Splitting the data into training and testing.
trainingSet = data(~split,:);
testingSet = data(split,:);

% Separating variable predictors and output values for training and test sets.
trainingPredictors = trainingSet(:,2:29);
trainingOutcomes = trainingSet{:,31};
testingPredictors = testingSet(:,2:29);
testingOutcomes = testingSet{:,31};

% Re-assining our training variables for legibility, the contents of X in array.
X = trainingPredictors;
y = trainingOutcomes;

%% Adding gaussian noise.

% We have commented this section out - as we found during the testing stage
% that our SVM did not react well with added noise to the data set (the
% accuracy reduced down to 56%). This is something to consider when
% evaluating this model and assessing its level of generalisability compared to MLP.

% Converting format of data for awgn suitability.
% X = table2array(X);
% testingPredictors = table2array(testingPredictors);

% Improving generalisation and preventing overfitting.
% X = awgn(X,8,'measured');
% testingPredictors = awgn(testingPredictors,8,'measured');

%% Preparing Cross-validation.

% Preparing cross-validation - provided size of training dataset.
c = cvpartition(836,'KFold',10);

%% Selecting optimised parameter values from Optimisation (ref:'Optimisation_SVM_Model').

% Our RBF kernel scale value (also known as sigma).
sigmaOpts = 6.8357;

% Our Box Constraint value.
boxOpts = 4.1815;

%% Performing SVM - training the model.

% Performing SVM and fitting the optimised parameters to the model (ref:'Optimisation_SVM_Model').
SVMModel = fitcsvm(X,y,'KernelFunction','rbf', 'KernelScale',sigmaOpts,...
    'BoxConstraint', boxOpts, 'ClassNames',{'0','1'}); % We use RBF kernel due to positive literature review findings.

% Calculating the K Fold Loss of the optimised model.
loss = kfoldLoss(fitcsvm(X,y,'CVPartition',c,...
    'KernelFunction','rbf','BoxConstraint',SVMModel.BoxConstraints(1),...
    'KernelScale',SVMModel.KernelParameters.Scale));

%% Predicting on the test set using our trained SVM data.

[predictedLabels,scores] = predict(SVMModel,testingPredictors); % Scores calculated to plot ROC curve.

%% Evaluate performance.

% Plotting confusion matrices.
figure;
predictedLabels = str2double(predictedLabels); % Converting predicted labels into double for accessibility.
confusionMatrix = plotconfusion(testingOutcomes',predictedLabels'); % Transposing data to fit into matrix.
[c,cm] = confusion(testingOutcomes',predictedLabels'); % Alternative function used to aid accuracy calculations.
title('SVM Confusion Matrix');
set(findobj(gca,'type','text'),'fontsize',20); % Changing font size.

% Calculating the accuracy.
compareTest = testingOutcomes == predictedLabels;
Accuracy = nnz(compareTest) / numel(testingOutcomes) * 100
 
% Calculating the recall.
Recall = cm(1,1)/(cm(1,1)+cm(1,2))
 
% Calculating the precision.
Precision = cm(1,1)/(cm(1,1)+cm(2,1))
 
% Calculating the specificity.
Specificity = cm(2,2)/(cm(2,1)+cm(2,2))

% Printing statements showing % of correct/incorrect classifications.
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

% ROC Curve - Using 'perfcurve' function.
[X,Y,T,AUC] = perfcurve(testingOutcomes',scores(:,1),'0','NegClass','1');
figure
plot(X,Y);
xlabel('False Positive Rate') 
ylabel('True Positive Rate')
title('SVM ROC Curve')

% ROC Curve - Using 'plotroc' function - used as a comparison measure to
% our MLP model to keep function consistency.
scores = scores(:,2); % Selecting the positive class scores (the second column).
plotroc(testingOutcomes',scores')% Plotting ROC curve.
title('SVM ROC Curve')

%% References

% https://uk.mathworks.com/help/stats/optimize-an-svm-classifier-fit-using-bayesian-optimization.html
% https://uk.mathworks.com/help/stats/support-vector-machines-for-binary-classification.html
% https://ww2.mathworks.cn/help/stats/bayesian-optimization-plot-functions.html

