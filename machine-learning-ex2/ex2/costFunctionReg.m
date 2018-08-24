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

% from new attempt
 % h = sigmoid(X*theta); % size h = 118 rows, 1 column 

% J = 1/m * ((log(h') * -y) - (log(1-h') * (1-y))) + (lambda/(2*m)*theta' * theta);

% grad(1) = 1/m * (h(1) - y(1))' * X(1);

% grad(2:end) = ((1/m * (h(2:end) - y(2:end))' * X(2:end))) + ((lambda/m) * theta);


% from previous attempt
h = sigmoid(X*theta); % size h = 118 rows, 1 column 

J_unreg = 1/m * ((log(h')*-y) - (log(1-h')*(1-y)));
theta(1) = 0; 
J_reg = (lambda/(2*m))*(theta'*theta);
J = J_unreg + J_reg; 

grad_unreg = ((h - y)'*X)/m;
grad_reg = theta*(lambda/m);
grad = grad_unreg' + grad_reg;

% =============================================================

end
