function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); % Number of training examples
deltaSum1 = 0; % Cumulative error between output layer and hidden layer 
deltaSum2 = 0; % Cumulative error between hidden layer and input layer

% Add ones to the X data matrix
X = [ones(m, 1) X];

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = X; % Input layer of neural network

% Iterate over all the training examples
for i = 1:m
    
    % Calculate z2 from input layer a1
    z2 = Theta1 * a1(i, :)';

    % Calculate hidden layer a2
    a2 = sigmoid(z2);

    % Add bias unit to hidden layer
    a2 = [1; a2];

    % Calculate z3 from hidden layer a2
    z3 = Theta2 * a2;

    % Calculate output layer
    a3 = sigmoid(z3);

    % Hypothesis/output layer is just a3
    h = a3;

    % Create zero initialized column vector for multi-class classification
    yClassifier = zeros(num_labels, 1);

    % Set the binary classifier where the training example output value
    % corresponds to an index into the yClassifier vector and is set to 1
    yClassifier(y(i, :)) = 1;

    % Calculate the running cost function
    J = J + sum((-yClassifier .* log(h)) - ((1.0 - yClassifier) .* log(1 - h)));

    % Calculate the backpropagation error delta for the output layer
    delta3(:, i) = a3 - yClassifier;

    % Calculate sigmoid gradient for the hidden layer
    gofz2prime = sigmoidGradient(z2);

    % Add bias unit
    gofz2prime = [1; gofz2prime];

    % Calculate the backpropagation error delta for the hidden layer
    delta2(:, i) = (Theta2' * delta3(:, i)) .* gofz2prime;
    
    % Accumulate the running delta for the output layer and hidden layer
    deltaSum2 = deltaSum2 + (delta3(:, i) * a2');
    deltaSum1 = deltaSum1 + (delta2(:, i) * a1(i, :));
end

% Calculate cost function
J = J / m;

% The cost function and gradient do not regularize on bias terms, 
% Theta1(:, 1) and Theta2(:, 1), both of which are the first columns of
% their respective matrices

% Remove the values from their vectors
Theta1(:, 1) = [];
Theta2(:, 1) = [];

% Calculate the regularized portion of the regularized cost function
regularizer = (lambda / (2.0 * m)) * ((sum(sum(Theta1 .^ 2)) + sum(sum(Theta2 .^ 2))));

% Add cost function to regularization
J = J + regularizer;

% Regularize the theta gradients
D2 = deltaSum2 / m;
D1 = deltaSum1 / m;

% The gradient does not regularize on the bias term, similar to above for
% cost function
D21 = D2(:, 1);
D2(:, 1) = [];

% Remove bias input terms here as well
D1(1, :) = [];
D11 = D1(:, 1);
D1(:, 1) = [];

% Continue regularization
D2 = D2 + (lambda / m) .* Theta2;
D1 = D1 + (lambda / m) .* Theta1;

% Recombine the bias term with the regularized gradient
Theta2_grad = [D21, D2];
Theta1_grad = [D11, D1];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
