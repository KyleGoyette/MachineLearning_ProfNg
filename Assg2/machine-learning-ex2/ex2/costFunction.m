function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
sumc=0;
psum=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
for i=1:m
    z=dot(theta,X(i,:));
    sumc=sumc+((-y(i)*log(sigmoid(z)))-(1-y(i))*(log(1-sigmoid(z))));
    for j=1:size(theta)
        grad(j)=grad(j)+(sigmoid(z)-y(i))*X(i,j);
    end
end


    
J=(1/m)*sumc;
grad=(1/m)*grad;

% J=(1/m)*(-1*log(sigmoid(X*theta))'*y-(log(1-sigmoid(X*theta))'*(1-y)))
% 
% grad=(1/m)*(sigmoid(X*theta)-y)'*X





% =============================================================

end
