function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
[m, n] = size(X); % number of training examples and features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Calculate the linear regression hypothesis function (vectorized)
h = X * theta;

% Calculate the cost function J for linear regression
J = sum((h - y) .^ 2) / (2.0 * m);

% Calculate the gradients for logistic regression (vectorized)
grad = X' * (h - y) / m;

% Save the grad value
grad0 = grad(1);

% Remove the values from their vectors
theta(1) = [];
grad(1) = [];

% Update cost function and gradient with regularized portion
J = J + (lambda * sum(theta .^ 2) / (2.0 * m));
grad = grad + (lambda / m) .* theta;

% Recombine the grad0 parameter with the regularized gradient
grad = [grad0; grad];

% =========================================================================

grad = grad(:);

end
