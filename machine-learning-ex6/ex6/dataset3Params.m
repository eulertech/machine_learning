function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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
err = 1.0E20;
params = [C*[0.01,0.1,1,10],sigma*[0.1,1,10,100]];
for i = 1 : length(params)
    for j = 1 : length(params)
        C_try = params(i);
        sigma_try = params(j);
        model= svmTrain(X, y, C_try, @(x1, x2) gaussianKernel(x1, x2, sigma_try));
        predictions = svmPredict(model,Xval);
        %error         
        if(mean(double(predictions ~=yval))<err)
           err = mean(double(predictions ~=yval)) ;
           C = C_try;
           sigma = sigma_try;
        end
        
        
    end
end





% =========================================================================

end
