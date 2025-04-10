% Task 2: Implement SVMs and Classify Data using Discriminant Functions
clc; clear all; close all;

%% Load Data
load('train.mat'); % Training data
load('test.mat');  % Test data

X_train = train_data; % 57x2000
y_train = train_label; % 2000x1
X_test = test_data; % 57x1536
y_test = test_label; % 1536x1

%% Data Normalization (Important for Stability)
X_train = X_train ./ max(abs(X_train), [], 2);
X_test = X_test ./ max(abs(X_test), [], 2);

%% Parameters
degrees = [2, 3, 4, 5];
C_values = [0.1, 0.6, 1.1, 2.1]; % For Soft-Margin
results = [];

%% Linear Kernel SVM
fprintf('\n--- Linear Kernel SVM ---\n');
[w_linear, b_linear] = linearSVM(X_train, y_train);
acc_train_linear = classifyLinearSVM(w_linear, b_linear, X_train, y_train);
acc_test_linear = classifyLinearSVM(w_linear, b_linear, X_test, y_test);
results = [results; {"Linear", "-", "-", acc_train_linear, acc_test_linear}];
fprintf('Linear Kernel - Training Accuracy: %.2f%% | Test Accuracy: %.2f%%\n', acc_train_linear, acc_test_linear);

%% Hard-Margin Polynomial Kernel SVM
for p = degrees
    fprintf('\n--- Hard-Margin Polynomial Kernel SVM (p = %d) ---\n', p);
    [Alpha, b_poly] = polynomialSVM(X_train, y_train, p, 1e6); % Hard-Margin using large C
    acc_train_poly = classifyKernelSVM(Alpha, b_poly, X_train, y_train, X_train, y_train, p);
    acc_test_poly = classifyKernelSVM(Alpha, b_poly, X_train, y_train, X_test, y_test, p);
    results = [results; {"Hard-Margin", p, "Inf", acc_train_poly, acc_test_poly}];
end

%% Soft-Margin Polynomial Kernel SVM
for p = degrees
    for C = C_values
        fprintf('\n--- Soft-Margin Polynomial Kernel SVM (p = %d, C = %.1f) ---\n', p, C);
        [Alpha, b_poly] = polynomialSVM(X_train, y_train, p, C);
        acc_train_poly = classifyKernelSVM(Alpha, b_poly, X_train, y_train, X_train, y_train, p);
        acc_test_poly = classifyKernelSVM(Alpha, b_poly, X_train, y_train, X_test, y_test, p);
        results = [results; {"Soft-Margin", p, C, acc_train_poly, acc_test_poly}];
    end
end

%% Display Results
result_table = cell2table(results, ...
    'VariableNames', {'SVM_Type', 'Degree_p', 'C', 'Training_Accuracy', 'Test_Accuracy'});
disp(result_table);

%% -------------------------
% FUNCTIONS
% -------------------------

% 1. Linear SVM using Hard-Margin
function [w, b] = linearSVM(X, y)
    H = (y * y') .* (X' * X);
    f = -ones(size(y));
    Aeq = y';
    Beq = 0;
    lb = zeros(size(y));
    ub = Inf(size(y));
    options = optimset('LargeScale', 'off', 'MaxIter', 50000, 'Display', 'iter');
    Alpha = quadprog(H, f, [], [], Aeq, Beq, lb, ub, [], options);
    w = X * (Alpha .* y);
    sv_idx = find(Alpha > 1e-4);
    b = mean(y(sv_idx) - X(:, sv_idx)' * w);
end

% 2. Polynomial Kernel SVM (Both Hard and Soft-Margin)
function [Alpha, b] = polynomialSVM(X, y, p, C)
    % Polynomial Kernel Matrix
    K_train = (X' * X + 1).^p;
    H = (y * y') .* K_train;
    f = -ones(size(y));
    Aeq = y';
    Beq = 0;
    lb = zeros(size(y));
    ub = ones(size(y)) * C;
    options = optimset('LargeScale', 'off', 'MaxIter', 50000, 'Display', 'iter');
    Alpha = quadprog(H, f, [], [], Aeq, Beq, lb, ub, [], options);
    
    % Calculate bias using support vectors
    sv_idx = find(Alpha > 1e-4);
    b = mean(y(sv_idx) - sum((Alpha .* y) .* K_train(:, sv_idx), 1)');
end

% 3. Classification for Linear SVM
function acc = classifyLinearSVM(w, b, X, y)
    predictions = sign(X' * w + b);
    acc = sum(predictions == y) / length(y) * 100;
end

% 4. Classification for Kernel SVM
function acc = classifyKernelSVM(Alpha, b, X_train, y_train, X_test, y_test, p)
    % Calculate the kernel matrix for the test set
    K_test = (X_test' * X_train + 1).^p;
    
    % Predict using discriminant function
    predictions = sign(K_test * (Alpha .* y_train) + b);
    
    % Accuracy Calculation
    acc = sum(predictions == y_test) / length(y_test) * 100;
end
