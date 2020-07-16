%% BAYESIAN OPTIMISATION FUNCTION %%

% We build this function separately.
% Here we are able to see the process of our parameter optimisation.  
% We evaluate the errors associated with the optimisation - ensuring
% that we continue to improve the validity and reliability of our model.
% We pass the optimised parameters found in the function below, through our
% SVM model in 'SVM_Model_Test'.

%% Importing the data

% Clearing previous command history.
clear; close all; clc

% Ensuring randomness of model remains constant.
% This means we are able to obtain the optimal results.
rng(0)
rng(4151941)
rng('default')

% Loading the data.
data = readtable('clean_data.csv');

% Removing unwanted column
data.Var1 = [];

%% Splitting the data set

% Splitting the data using randperm.
split_size = 0.15;
N = size(data,1);
split = false(N,1);
split(1:round(split_size*N)) = true;
split = split(randperm(N)); % Using randperm to reduce bias.

% Splitting data into training and testing sets.
trainingSet = data(~split,:);
testingSet = data(split,:);

% Separated variable predictors and output values for training and test set.
trainingPredictors = trainingSet(:,2:29);
trainingOutcomes = trainingSet{:,31};
testingPredictors = testingSet(:,2:29);
testingOutcomes = testingSet{:,31};

% Re-assining our training variables for legibility, the contents of X in array.
X = trainingPredictors;
y = trainingOutcomes;

%% Preparing Cross-validation

% Preparing cross-validation - provided size of training dataset.
c = cvpartition(836,'KFold',10);

%% Preparing Variables for Optimisation

% We firstly choose a wide range of values for sigma and box, and begin to reduce them as we
% find a suitable threshold that yield positive results.
sigma = optimizableVariable('sigma',[1 10],'Transform','log');
box = optimizableVariable('box',[1 10],'Transform','log');

% Setting and calling bayesian optimisation plot function - searching for the best parameters [sigma, box].
% For reproducibility, we specify the 'expected-improvement-plus' acquisition function. 
obj = @(x)mysvmminfn(x,X,y,c);
results = bayesopt(obj,[sigma,box],...
    'IsObjectiveDeterministic',true,...
    'AcquisitionFunctionName','expected-improvement-plus',...
    'PlotFcn',{@supportVec,@plotObjectiveModel,@plotMinObjective}); % referencing the support vector plot function.

%% Computing cross-validation loss & computing SVM Vectors
% Creating an objective function that computes the cross-validation loss for a fixed cross-validation partition, 
% and that returns the number of support vectors in the resulting model.

function [f,v,nsupp] = mysvmminfn(x,X,y,c)
SVMModel = fitcsvm(X,y,'KernelFunction','rbf',...
    'KernelScale',x.sigma,'BoxConstraint',x.box);
f = kfoldLoss(crossval(SVMModel,'CVPartition',c));
v = [];
nsupp = sum(SVMModel.IsSupportVector);
end

%% Custom Plot Function 
% Plotting both the current number of constraints and the number of constraints for the model 
% with the best objective function found.
% We use information in the UserData property of the BayesianOptimization object.
% This model was built with help from MATLAB mathworks - https://ww2.mathworks.cn/help/stats/bayesian-optimization-plot-functions.html

function i = supportVec(results,state)
persistent hs nbest besthist nsupptrace
i = false;
switch state
    
    case 'initial'
        hs = figure;
        besthist = [];
        nbest = 0;
        nsupptrace = [];
    case 'iteration'
        
        figure(hs)
        nsupp = results.UserDataTrace{end};   % retrieving nsupp from UserDataTrace property.
        nsupptrace(end+1) = nsupp; % accumulating nsupp values in a vector.
        if (results.ObjectiveTrace(end) == min(results.ObjectiveTrace)) || (length(results.ObjectiveTrace) == 1)
            nbest = nsupp;
        end
        
        besthist = [besthist,nbest];
        plot(1:length(nsupptrace),nsupptrace,'b',1:length(besthist),besthist,'r--')
        xlabel('Iteration number')
        ylabel('Number of support vectors')
        title('Number of support vectors at each iteration')
        legend('Current iteration','Best objective','Location','best')
        drawnow
        
end
end

%% References

% https://uk.mathworks.com/help/stats/optimize-an-svm-classifier-fit-using-bayesian-optimization.html
% https://uk.mathworks.com/help/stats/support-vector-machines-for-binary-classification.html
% https://ww2.mathworks.cn/help/stats/bayesian-optimization-plot-functions.html

