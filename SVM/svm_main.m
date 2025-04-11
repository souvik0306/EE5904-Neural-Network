% svm_main.m - Task 3: Predict eval_predicted using best polynomial SVM

% Load training data
load('train.mat'); % train_data [57x2000], train_label [2000x1]
X_train = train_data;
y_train = train_label;

% Load evaluation data
load('eval.mat'); % eval_data [57x600]
X_eval = eval_data;

% === Standardize using training statistics ===
mean_train = mean(X_train, 2);
std_train = std(X_train, 0, 2) + 1e-8;
X_train = (X_train - mean_train) ./ std_train;
X_eval  = (X_eval  - mean_train) ./ std_train;

% === Polynomial Kernel Parameters ===
p = 5;         % Polynomial degree
C = 2.1;       % Soft-margin parameter

% === Compute Normalized Polynomial Kernel (Train) ===
dot_prod = X_train' * X_train;
dot_prod = dot_prod ./ max(abs(dot_prod(:)));  % Normalize to stabilize
K_train = (dot_prod + 1).^p;

% === QP Setup ===
n = size(X_train, 2);
H = (y_train * y_train') .* K_train;
f = -ones(n, 1);
Aeq = y_train';
beq = 0;
lb = zeros(n, 1);
ub = C * ones(n, 1);
options = optimset('LargeScale', 'off', 'MaxIter', 10000, 'Display', 'off');

% === Solve QP ===
Alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);

% === Compute Bias ===
sv_idx = find(Alpha > 1e-4);
b = mean(y_train(sv_idx) - K_train(sv_idx, :) * (Alpha .* y_train));

% === Compute Normalized Polynomial Kernel (Eval) ===
dot_prod_eval = X_eval' * X_train;
dot_prod_eval = dot_prod_eval ./ max(abs(dot_prod(:))); % Use same scale
K_eval = (dot_prod_eval + 1).^p;

% === Predict ===
eval_predicted = sign(K_eval * (Alpha .* y_train) + b);
eval_predicted = eval_predicted(:); % Ensure column vector [600 x 1]

fprintf('Evaluation complete. Output: eval_predicted [%d x 1]\n', length(eval_predicted));
