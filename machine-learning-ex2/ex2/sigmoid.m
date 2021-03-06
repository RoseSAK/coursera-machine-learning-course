function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Vectorized implementation
g = 1./(1+(exp(-z))); 
% g = (1./(1+ (e.^-z))); - alternative

% Previous unvectorized implementation
%for i = 1:length(g)
 % g(i) = 1/(1+e^-z(i));
%end

% =============================================================

end
