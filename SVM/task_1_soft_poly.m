% Task 1: Soft-Margin SVM with Polynomial Kernel using Quadratic Programming
clearvars; clc;

fprintf('=== Task 1: Soft-Margin Polynomial SVM Training Only ===\n');

%% Load Training Data
fprintf('[1/5] Loading training data...\n');
load('train.mat'); % train_data: [57x2000], train_label: [2000x1]
X_train = train_data;
y_train = train_label;

%% Standardization (using training stats only)
fprintf('[2/5] Standardizing features...\n');
mean_train = mean(X_train, 2);
std_train = std(X_train, 0, 2) + 1e-8;
X_train = (X_train - mean_train) ./ std_train;

%% Prepare QP Setup
n_samples = size(X_train, 2);
A = []; b = [];
Aeq = y_train';
Beq = 0;
lb = zeros(n_samples, 1);
f = -ones(n_samples, 1);

% Values for C and p (as per project Table 1)
C_values = [0.1, 0.6, 1.1, 2.1];
degrees = [1, 2, 3, 4, 5];
results = zeros(length(degrees) * length(C_values), 3);

row_index = 1;

%% Start Training
for i = 1:length(degrees)
    p = degrees(i);
    for j = 1:length(C_values)
        C = C_values(j);
        fprintf('\n[3/5] Training SVM (p = %d, C = %.1f)...\n', p, C);
        
        % Normalize dot products before applying polynomial kernel
        dot_prod = X_train' * X_train;
        dot_prod = dot_prod ./ max(abs(dot_prod(:)));
        K = (dot_prod + 1).^p;
        H = (y_train * y_train') .* K;
        ub = C * ones(n_samples, 1); % Soft margin bounds

        % Solve QP
        options = optimset('LargeScale', 'off', 'MaxIter', 10000, 'Display', 'off');
        Alpha = quadprog(H, f, A, b, Aeq, Beq, lb, ub, [], options);

        % Support Vectors
        idx = find(Alpha > 1e-4);

        % Compute Bias b
        b_poly = mean(y_train(idx) - K(idx, :) * (Alpha .* y_train));

        % Predict on training data
        y_pred_train = sign(K * (Alpha .* y_train) + b_poly);
        acc_train = mean(y_pred_train == y_train) * 100;

        fprintf('Training Accuracy (p=%d, C=%.1f): %.2f%%\n', p, C, acc_train);

        % Store result
        results(row_index, :) = [p, C, acc_train];
        row_index = row_index + 1;
    end
end

%% Display Final Table
fprintf('\n[5/5] Summary Table (Training Accuracies Only):\n');
results_table = array2table(results, ...
    'VariableNames', {'Degree_p', 'C', 'Training_Accuracy'});
disp(results_table);
