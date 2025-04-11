% Improved Linear Kernel SVM using Quadratic Programming
clc; clear all; close all;

%% Load Data
load('train.mat'); % Training data
load('test.mat');  % Test data

X_train = train_data; % 57x2000
y_train = train_label; % 2000x1
X_test = test_data; % 57x1536
y_test = test_label; % 1536x1

%% =======================
% Feature Normalization
% Normalize each feature (row-wise) using training data stats
mean_train = mean(X_train, 2);
std_train = std(X_train, 0, 2) + 1e-8; % small value to avoid division by zero

X_train = (X_train - mean_train) ./ std_train;
X_test = (X_test - mean_train) ./ std_train; % Use training stats to normalize test set
%% =======================

%% Set Parameters
A = [];
b = [];
Aeq = y_train';
Beq = 0;

n_samples = size(X_train, 2);
lb = zeros(n_samples, 1);
C = 1e6; % Large C for Hard-Margin
ub = ones(n_samples, 1) * C;
f = -ones(n_samples, 1);

%% Compute Kernel Matrix (Linear Kernel)
H_sign = y_train * y_train'; % 2000x2000
H = (X_train' * X_train) .* H_sign; % 2000x2000

%% Solve Using Quadratic Programming
options = optimset('LargeScale', 'off', 'MaxIter', 10000, 'Display', 'iter');
Alpha = quadprog(H, f, A, b, Aeq, Beq, lb, ub, [], options);

%% Calculate Discriminant Function Parameters
% Support Vectors Identification
idx = find(Alpha > 1e-4);

% Compute w and b
wo = X_train * (Alpha .* y_train); % 57x1
bo = mean(y_train(idx) - (X_train(:, idx)' * wo));

fprintf('Linear Kernel SVM Results:\n');
fprintf('w = [%s]\n', num2str(wo'));
fprintf('b = %.4f\n', bo);

%% Evaluate Accuracy
acc_train = Accuracy(wo, bo, X_train, y_train);
acc_test = Accuracy(wo, bo, X_test, y_test);
fprintf('Training Accuracy: %.2f%%\n', acc_train * 100);
fprintf('Test Accuracy: %.2f%%\n', acc_test * 100);

%% Accuracy Calculation Function
function acc = Accuracy(w, b, X, y)
    predictions = sign(X' * w + b);
    acc = sum(predictions == y) / length(y);
end
