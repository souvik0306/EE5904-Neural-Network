% Hard-Margin SVM with Polynomial Kernel using Quadratic Programming (Improved)
clc; clear; close all;

%% Load Data
load('train.mat'); % Training data
load('test.mat');  % Test data

X_train = train_data; % 57x2000
y_train = train_label; % 2000x1
X_test = test_data; % 57x1536
y_test = test_label; % 1536x1

%% Data Normalization
X_train = X_train ./ max(abs(X_train), [], 2);
X_test = X_test ./ max(abs(X_test), [], 2);

%% Parameters
A = [];
b = [];
Aeq = y_train';
Beq = 0;

n_samples = size(X_train, 2);
lb = zeros(n_samples, 1);
C = 10; % Adjusting for stability
ub = ones(n_samples, 1) * C;
f = -ones(n_samples, 1);

%% Polynomial Kernel Implementation
degrees = [2, 3, 4, 5];
results = zeros(length(degrees), 2);

for i = 1:length(degrees)
    p = degrees(i);
    fprintf('\nTraining Hard-Margin SVM with Polynomial Kernel (p = %d)...\n', p);
    
    % Polynomial Kernel Calculation
    K_train = (X_train' * X_train + 1).^p;
    H_sign = y_train * y_train';
    H = K_train .* H_sign;

    %% Solve Using Quadratic Programming
    options = optimset('LargeScale', 'off', 'MaxIter', 10000, 'Display', 'iter');
    Alpha = quadprog(H, f, A, b, Aeq, Beq, lb, ub, [], options);

    %% Calculate Discriminant Function Parameters
    % Identify Support Vectors
    idx = find(Alpha > 1e-3);

    % Compute b using reliable support vectors
    b_poly = mean(y_train(idx) - sum((Alpha .* y_train) .* K_train(:, idx), 1)');

    % Polynomial Kernel for Test Data
    K_test = (X_test' * X_train + 1).^p;

    % Predict using Polynomial Kernel SVM
    y_train_pred = sign((K_train * (Alpha .* y_train)) + b_poly);
    y_test_pred = sign((K_test * (Alpha .* y_train)) + b_poly);

    %% Calculate Accuracy
    acc_train = sum(y_train_pred == y_train) / length(y_train) * 100;
    acc_test = sum(y_test_pred == y_test) / length(y_test) * 100;
    fprintf('Training Accuracy (p=%d): %.2f%%\n', p, acc_train);
    fprintf('Test Accuracy (p=%d): %.2f%%\n', p, acc_test);
    
    % Store results for table
    results(i, :) = [acc_train, acc_test];
end

%% Display Results Table
fprintf('\nFinal Results Table:\n');
disp(array2table(results, 'VariableNames', {'Training_Accuracy', 'Test_Accuracy'}, ...
    'RowNames', {'p=2', 'p=3', 'p=4', 'p=5'}));
