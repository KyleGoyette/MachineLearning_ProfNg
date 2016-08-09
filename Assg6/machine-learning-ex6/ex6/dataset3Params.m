function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_ind = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_ind = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
error=zeros(length(C_ind),length(sigma_ind));
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

for i=1:length(C_ind)
    for j=1:length(sigma_ind)
        Ci=C_ind(i);
        sig=sigma_ind(j);
        model = svmTrain(X, y, Ci, @(x1, x2)gaussianKernel(x1,x2,sig));
        pred=svmPredict(model,Xval);
        error(i,j)=mean(double(pred~=yval));
    end
end

[M,I]=min(error(:));

[I_row,I_col]=ind2sub(size(error),I);
C=C_ind(I_row);
sigma=sigma_ind(I_col);




% =========================================================================

end
