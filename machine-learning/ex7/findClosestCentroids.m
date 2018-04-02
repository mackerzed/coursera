function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Some useful values
K = size(centroids, 1); % Number of centroids
m = size(X, 1); % Number of training examples
cVec = ones(K, 1); % Used for vectorized computation of training examples

% You need to return the following variables correctly.
idx(1:size(X,1)) = Inf;

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Iterate over each centroid in every training example
for i = 1:m

    % Create anonymous vectorized normal function
    vecnormal = @(A) sqrt(sum(A .^ 2, 2));

    % Convert training example into vectorized training example matching
    % the dimensions of the centroids vector
    Xvec = cVec .* X(i, :);

    % Calculate normal of training example for each centroid (vectorized)
    normVec = vecnormal(Xvec - centroids) .^ 2;

    % Find the min value in the resultant normal vector
    [c, minIdx] = min(normVec);

    % Set the centroid corresponding centroid to this training example
    idx(i) = minIdx;
end

% =============================================================

end

