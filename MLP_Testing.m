%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%  Multi Layer Perceptron Model  %%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% See 'Optimisation_MLP_Model.m' to see how we optimised our model and for
% more technical details.

%% Importing the data.

% Clearing previous command history.
clear; close all; clc

% Ensuring randomness remains constant.
% This means we are able to obtain the optimal results.
rng('default')
 
% Loading the data.
data = readtable('clean_data.csv');

% Removing unwanted column.
data.Var1 = [];

%% Splitting the data set.

% Assigning input and target data.
inputs = data{:,2:29};
targets = data{:,31};

% Normalising data.
normalisedInput = normalize(inputs);

% Adding gaussian noise to data to prevent overfitting and improve generalisation.
inputs = awgn(normalisedInput,8,'measured');

% Re-assigning input and target data to x and t.
x = inputs';
t = targets';

%% Training MLP Model

% Using 'trainscg' function - it's faster, uses less memory & is suitable in low memory situations.
trainFcn = 'trainscg';
net = patternnet(12, trainFcn); % 1 hidden layer 12 neurons.

% Random set-up division of data for training, validation, testing.
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Setting optimised training parameters found using bayesian optimisation (see Optimisation_MLP_Model.m).
% We have already set the size of hidden layers in line 45.
net.trainParam.mc=0.81969; % The Momentum

% Setting the other parameters to aid understanding of model configuration.
% We vary these depending on our outcomes. Reasons for these training
% parameter values are explained in our report.
net.trainParam.goal=0; % The error goal.
net.trainParam.epochs=100; % The maximum iterations.
net.trainParam.show=25; % Showing the intervals.
net.trainParam.max_fail=30; % Maximum failures - this affects the validation.
net.trainParam.sigma=5.0e-5; % Sigma - change in weight for second derivative approximation.
net.trainParam.lambda=5.0e-7; % Lambda - parameter for regulating the indefiniteness of the Hessian.

% Training the final network using optimised parameters.
[net,tr] = train(net,x,t);
nntraintool
plotperform(tr)

%% Testing the network on test set

% Testing the network using optimised parameters.
testX = x(:,tr.testInd);
testT = t(:,tr.testInd);
testY = net(testX);
testIndices = vec2ind(testY);

%% Evaluating Performance

% Plotting confusion matrix.
figure;
plotconfusion(testT,testY); % Using test data.
set(findobj(gca,'type','text'),'fontsize',20); % Changing font size.
[c,cm] = confusion(testT,testY);
title('MLP Confusion Matrix'); % Setting title for figure.

%Calculating the accuracy score using confusion matrix results.
Accuracy = 100*sum(diag(cm))./sum(cm(:))
 
%Calculating the recall score.
Recall = cm(1,1)/(cm(1,1)+cm(1,2))
 
%Calculating the precision score.
Precision = cm(1,1)/(cm(1,1)+cm(2,1))
 
%Calculating the specificity score.
Specificity = cm(2,2)/(cm(2,1)+cm(2,2))

% Printing statements showing % of correct/incorrect classifications.
fprintf('Percentage of Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage of Incorrect Classification : %f%%\n', 100*c);

% Plotting the ROC curve.
plotroc(testT,testY) % Using test data.
title('MLP ROC Curve')

%% References

% MATLAB Neural Network Toolbox
% https://uk.mathworks.com/help/deeplearning/ug/deep-learning-using-bayesian-optimization.html
