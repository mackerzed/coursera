function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

% Use costFunction to calculate the cost function and gradient
[J, grad] = costFunction(theta, X, y);

% The cost function and gradient do not regularize on theta0 nor grad0,
% respectively, both of which are the first entries of their respective
% vectors

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

end
