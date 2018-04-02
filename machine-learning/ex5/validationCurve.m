function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Some useful variables
m = size(X, 1); % Number of training examples
mcv = size(Xval, 1); % Number of cross validation examples

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

% Iterate over the vector of lambdas to find the training and validation
% errors for each lambda
for i = 1:length(lambda_vec)
    % Train dataset with given lambda
    theta = trainLinearReg(X, y, lambda_vec(i));

    % Calculate hypothesis with calculated theta on training set
    hTrain = X * theta;

    % Calculate training error for hypothesis hTrain
    error_train(i) = sum((hTrain - y) .^ 2) / (2.0 * m);

    % Calculate hypothesis with calculated theta on cross validation set
    hVal = Xval * theta;

    % Calculate training error for hypothesis hVal
    error_val(i) = sum((hVal - yval) .^ 2) / (2.0 * mcv);
end

% =========================================================================

end
