function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Setup some useful variables
values_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; % vector of C and sigma values to test
bestC = values_vec(1); % Initialize
bestSigma = values_vec(1); % Initialize
bestError = 1; % Initialize to 100% error; try to find a better (lower) error

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Check every combination of C and sigma from the values to test
for i = 1:length(values_vec)
    C = values_vec(i);
    for j = 1:length(values_vec)
        sigma = values_vec(j);

        % Train the model with the gaussian kernel
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

        % Make predictions on the cross validation set
        predictions = svmPredict(model, Xval);

        % Calculate the error of the model on the cross validation set
        error = mean(double(predictions ~= yval));

        % Use the best C and best sigma when we find the lowest error
        if (error < bestError)
            bestError = error;
            bestC = C;
            bestSigma = sigma;
        end
    end
end

% Assign output values
C = bestC;
sigma = bestSigma;

% =========================================================================

end
