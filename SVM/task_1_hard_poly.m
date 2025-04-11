% Task 1: Hard-Margin SVM with Polynomial Kernel using Quadratic Programming
clearvars; clc;

fprintf('=== Task 1: Hard-Margin SVM with Polynomial Kernel ===\n');

%% Load Training Data
fprintf('[1/5] Loading training data...\n');
load('train.mat'); % train_data: [57x2000], train_label: [2000x1]
X_train = train_data;
y_train = train_label;

%% Feature Standardization (Training Stats Only)
fprintf('[2/5] Standardizing features...\n');
mean_train = mean(X_train, 2);
std_train = std(X_train, 0, 2) + 1e-8;
X_train = (X_train - mean_train) ./ std_train;

%% Setup QP Parameters
fprintf('[3/5] Preparing QP problem...\n');
n_samples = size(X_train, 2);
A = []; b = [];
Aeq = y_train';
Beq = 0;
lb = zeros(n_samples, 1);
C = 1e6; % Hard-margin SVM approximated with large C
ub = C * ones(n_samples, 1);
f = -ones(n_samples, 1);

%% Polynomial Kernel Loop
degrees = [2, 3, 4, 5];
results = zeros(length(degrees), 1);

for i = 1:length(degrees)
    p = degrees(i);
    fprintf('\n[4/5] Training Polynomial Kernel SVM (p = %d)...\n', p);
    
    % Compute Gram Matrix with Polynomial Kernel
    dot_prod = X_train' * X_train;
    dot_prod = dot_prod ./ max(abs(dot_prod(:)));  % scale inner products
    K = (dot_prod + 1).^p;
    H = (y_train * y_train') .* K;

    % Solve QP
    options = optimset('LargeScale', 'off', 'MaxIter', 10000, 'Display', 'off');
    Alpha = quadprog(H, f, A, b, Aeq, Beq, lb, ub, [], options);

    % Support Vector Detection
    idx = find(Alpha > 1e-3);

    % Compute Bias
    b_poly = mean(y_train(idx) - K(idx, :) * (Alpha .* y_train));

    % Predict on Training Data
    y_pred_train = sign(K * (Alpha .* y_train) + b_poly);
    acc_train = mean(y_pred_train == y_train) * 100;
    fprintf('Training Accuracy (p = %d): %.2f%%\n', p, acc_train);

    results(i) = acc_train;
end

%% Final Results
fprintf('\n[5/5] Summary Table (Training Accuracies Only):\n');
disp(array2table(results, ...
    'VariableNames', {'Training_Accuracy'}, ...
    'RowNames', {'p=2', 'p=3', 'p=4', 'p=5'}));
