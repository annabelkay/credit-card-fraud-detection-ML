%% BAYESIAN OPTIMISATION FUNCTION %%

% RUN MLP_TESTING FIRST

% We build this function separately.
% Here we are able to see the process of our parameter optimisation.  
% We evaluate the errors associated with the optimisation - ensuring
% that we continue to improve the validity and reliability of our model.
% We pass the optimised parameters found in the function below, through our
% MLP model in 'MLP_Testing'.

%% Importing the Data

% Ensuring randomness remains constant.
rng('default')

% Assigning new variable names to inputs and targets.
Inputs = inputs';
Targets = targets';

%% Preparing the Optimisation Function (see bottom).

% Choosing our variables to optimise: layer size & momentum.
optimVars = [optimizableVariable('LayerSize',[10 15],'Type','integer') % 10-15 hidden neurons.
    optimizableVariable('Momentum',[0.6 0.89])];

% Performing our bayesian optimisation function.
% Iterating through the model and assigning objects/parameters.
ObjFcn = makeObjFcn(Inputs, Targets);
BayesObject = bayesopt(ObjFcn,optimVars,...
    'MaxObj',30,...
    'MaxTime',8*60*60,...
    'IsObjectiveDeterministic',false,...
    'UseParallel',false);

%% Evaluating the Final Network.

% We use a test error and validation error score to evaluate the accuracy
% and reliability of our model for real-life credit card fraud detection scenarios.
bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
load(fileName);

% Calculating test and validation errors.
YPredicted = net(Inputs);
testError = perform(net,Targets,YPredicted);
testError;
valError;

%% Objective Function for Bayesian Optimisation.

% Function runs through line 27.
function ObjFcn = makeObjFcn(XTrain,YTrain) % Using training and validation data.
    ObjFcn = @valErrorFun;
      function [valError,cons,fileName] = valErrorFun(optVars) % Building the function around parameters to be optimised.
          
          % Choosing a Training Function
          % 'trainscg' uses less memory & is suitable in low memory situations.
          trainFcn = 'trainscg'; 
          
          % Creating a Fitting Network
          layer_size = optVars.LayerSize;
          hiddenLayerSize = (layer_size);
          net = fitnet(hiddenLayerSize,trainFcn);
         
          % Setting Up Random Division of Data for Training, Validation and Testing.
          net.divideParam.trainRatio = 70/100; % For optimising the error gradient.
          net.divideParam.valRatio = 15/100; % For measuring generalisation and preventing overfitting.
          net.divideParam.testRatio = 15/100;% For an independent test.
          
          % Training the Network.
          net.trainParam.showWindow = false;
          net.trainParam.showCommandLine = false;
          [net,~] = train(net,XTrain,YTrain);
         
          % Testing the Network.
          YPredicted = net(XTrain);
          valError = perform(net,YTrain,YPredicted);
          fileName = num2str(valError) + ".mat";
          save(fileName,'net','valError')
          cons = [];
      end
  end

%% References

% MATLAB Neural Network Toolbox
% https://uk.mathworks.com/help/deeplearning/ug/deep-learning-using-bayesian-optimization.html


