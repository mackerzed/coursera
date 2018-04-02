function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features
J_history = zeros(num_iters, 1); % Cost function iteration history
thetaNew = zeros(n, 1); % temporary to store updates to theta

% Perform num_iters iterations of gradient descent
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % Calculate vectorized hypothesis h
    h = X * theta;

    % Iterate over all features and update corresponding theta
    for i = 1:n
        thetaNew(i, :) = theta(i, :) - (alpha / m) * sum((h - y) .* X(:, i));
    end

    % Update theta
    theta = thetaNew;

    % ============================================================

    % Save the cost J in every iteration using newly calculated theta
    J_history(iter) = computeCost(X, y, theta);

end

end
