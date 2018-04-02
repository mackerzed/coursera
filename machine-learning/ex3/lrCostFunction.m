function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% Calculate the logistic regression hypothesis function (vectorized)
h = sigmoid(X * theta);

% Calculate the cost function for logistic regression
J = sum((-y .* log(h)) - ((1.0 - y) .* log(1 - h))) / m;

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

% =============================================================

grad = grad(:);

end
