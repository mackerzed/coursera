function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% gradientDescent contians support for batch gradient descent with multiple
% variables/features
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

end
