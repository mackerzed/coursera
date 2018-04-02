function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Calculate hypothesis for cost function
h = X * Theta';

% Calculate cost function
J = sum(sum((R .* (h - Y)) .^ 2)) / 2;

% Update cost function with regularization
J = J + ((lambda  / 2.0) * (sum(sum(Theta .^ 2)) + sum(sum(X .^ 2))));

% Calculate the X gradients
for i = 1:num_movies
    % Find the indices of the users who have rated movie i
    idx = find(R(i, :) == 1);

    % Select the Theta and Y components corresponding to R(i, j) = 1 (ie.
    % the users who have rated movie i)
    Theta_ij = Theta(idx, :);
    Y_ij = Y(i, idx);

    % Calculate the hypothesis
    h = X(i, :) * Theta_ij';

    % Calculate the X gradient
    X_grad_i = (h - Y_ij) * Theta_ij;

    % Add the regularization term
    X_grad(i, :) = X_grad_i + (lambda .* X(i, :)) 
end

% Calculate the Theta gradients
for j = 1:num_users
    % Find the indices of the movies that have been rated by user j
    idx = find(R(:, j) == 1);

    % Select the X and Y components corresponding to R(i, j) = 1 (ie.
    % the movies that have been rated by user j)
    X_i = X(idx, :);
    Y_ij = Y(idx, j);

    % Calculate the hypothesis
    h = X_i * Theta(j, :)';

    % Calculate the Theta gradient
    Theta_grad_j = (h - Y_ij)' * X_i;

    % Add the regularization term
    Theta_grad(j, :) = Theta_grad_j + (lambda .* Theta(j, :));
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
