% Task 2: Classify Train and Test Sets using Trained SVMs
clearvars; clc;

%% Load Data
fprintf('Loading data...\n');
load('train.mat'); load('test.mat');
X_train = train_data; y_train = train_label;
X_test = test_data; y_test = test_label;

%% Standardize using training stats only
fprintf('Standardizing features...\n');
mean_train = mean(X_train, 2);
std_train = std(X_train, 0, 2) + 1e-8;
X_train = (X_train - mean_train) ./ std_train;
X_test  = (X_test  - mean_train) ./ std_train;

%% Set parameters
degrees = [2, 3, 4, 5];
C_values = [0.1, 0.6, 1.1, 2.1];
results = [];

%% Linear Kernel SVM (Hard Margin)
fprintf('\n--- Linear Kernel SVM ---\n');
[w_linear, b_linear] = linearSVM(X_train, y_train);
acc_train_linear = classifyLinearSVM(w_linear, b_linear, X_train, y_train);
acc_test_linear = classifyLinearSVM(w_linear, b_linear, X_test, y_test);
results = [results; {"Linear", "-", "-", acc_train_linear, acc_test_linear}];

%% Hard-Margin Polynomial Kernel SVM
for p = degrees
    fprintf('\n--- Hard-Margin Polynomial Kernel SVM (p = %d) ---\n', p);
    [Alpha, b_poly] = polynomialSVM(X_train, y_train, p, 1e6); % Large C = hard-margin
    acc_train = classifyKernelSVM(Alpha, b_poly, X_train, y_train, X_train, y_train, p);
    acc_test  = classifyKernelSVM(Alpha, b_poly, X_train, y_train, X_test, y_test, p);
    results = [results; {"Hard-Margin", p, "Inf", acc_train, acc_test}];
end

%% Soft-Margin Polynomial Kernel SVM
for p = degrees
    for C = C_values
        fprintf('\n--- Soft-Margin Polynomial Kernel SVM (p = %d, C = %.1f) ---\n', p, C);
        [Alpha, b_poly] = polynomialSVM(X_train, y_train, p, C);
        acc_train = classifyKernelSVM(Alpha, b_poly, X_train, y_train, X_train, y_train, p);
        acc_test  = classifyKernelSVM(Alpha, b_poly, X_train, y_train, X_test, y_test, p);
        results = [results; {"Soft-Margin", p, C, acc_train, acc_test}];
    end
end

%% Display Results
fprintf('\nFinal Classification Results:\n');
result_table = cell2table(results, ...
    'VariableNames', {'SVM_Type', 'Degree_p', 'C', 'Training_Accuracy', 'Test_Accuracy'});
disp(result_table);

%% -------------------------
% FUNCTIONS
% -------------------------

function [w, b] = linearSVM(X, y)
    H = (y * y') .* (X' * X);
    f = -ones(size(y));
    Aeq = y'; Beq = 0;
    lb = zeros(size(y)); ub = 1e6 * ones(size(y)); % large C ~ hard-margin
    options = optimset('LargeScale', 'off', 'MaxIter', 10000, 'Display', 'off');
    Alpha = quadprog(H, f, [], [], Aeq, Beq, lb, ub, [], options);
    w = X * (Alpha .* y);
    sv_idx = find(Alpha > 1e-4);
    b = mean(y(sv_idx) - X(:, sv_idx)' * w);
end

function [Alpha, b] = polynomialSVM(X, y, p, C)
    dot_prod = X' * X;
    dot_prod = dot_prod ./ max(abs(dot_prod(:)));  % normalize inner products
    K = (dot_prod + 1).^p;
    H = (y * y') .* K;
    f = -ones(size(y));
    Aeq = y'; Beq = 0;
    lb = zeros(size(y)); ub = ones(size(y)) * C;
    options = optimset('LargeScale', 'off', 'MaxIter', 10000, 'Display', 'off');
    Alpha = quadprog(H, f, [], [], Aeq, Beq, lb, ub, [], options);
    sv_idx = find(Alpha > 1e-4);
    b = mean(y(sv_idx) - K(sv_idx, :) * (Alpha .* y));
end

function acc = classifyLinearSVM(w, b, X, y)
    predictions = sign(X' * w + b);
    acc = mean(predictions == y) * 100;
end

function acc = classifyKernelSVM(Alpha, b, X_train, y_train, X_test, y_test, p)
    dot_prod = X_test' * X_train;
    dot_prod = dot_prod ./ max(abs(dot_prod(:)));
    K_test = (dot_prod + 1).^p;
    predictions = sign(K_test * (Alpha .* y_train) + b);
    acc = mean(predictions == y_test) * 100;
end
