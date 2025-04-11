% svm_main.m - Task 3: Predict eval_predicted using RBF Kernel SVM

% Load training data
load('train.mat'); % train_data [57x2000], train_label [2000x1]
X_train = train_data;
y_train = train_label;

% Load evaluation data (provided at runtime)
load('eval.mat'); % eval_data [57x600]
X_eval = eval_data;

% === Standardize using training statistics ===
mean_train = mean(X_train, 2);
std_train = std(X_train, 0, 2) + 1e-8;
X_train = (X_train - mean_train) ./ std_train;
X_eval  = (X_eval  - mean_train) ./ std_train;

% === RBF Kernel Parameters ===
C = 1.0;         % Soft-margin regularization
gamma = 0.05;    % RBF kernel width

% === Compute RBF Kernel (Train) ===
X1 = X_train; X2 = X_train;
K_train = exp(-gamma * (sum(X1.^2,1)' + sum(X2.^2,1) - 2*(X1'*X2)));

% === Solve QP ===
n = size(X_train, 2);
H = (y_train * y_train') .* K_train;
f = -ones(n, 1);
Aeq = y_train';
beq = 0;
lb = zeros(n, 1);
ub = C * ones(n, 1);
options = optimset('LargeScale', 'off', 'MaxIter', 10000, 'Display', 'off');
Alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);

% === Compute Bias ===
sv_idx = find(Alpha > 1e-4);
b = mean(y_train(sv_idx) - K_train(sv_idx, :) * (Alpha .* y_train));

% === Compute RBF Kernel (Eval) ===
X1 = X_eval; X2 = X_train;
K_eval = exp(-gamma * (sum(X1.^2,1)' + sum(X2.^2,1) - 2*(X1'*X2)));

% === Predict ===
eval_predicted = sign(K_eval * (Alpha .* y_train) + b);
eval_predicted = eval_predicted(:); % Ensure 600 x 1

fprintf('Evaluation complete. Output: eval_predicted [%d x 1]\n', length(eval_predicted));
