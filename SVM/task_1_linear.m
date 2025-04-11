% Hard-Margin SVM with Linear Kernel (Task 1) + Progress Messages
clearvars; clc;

fprintf('=== Task 1: Training Hard-Margin SVM with Linear Kernel ===\n');

%% Load Training Data
fprintf('[1/5] Loading training data...\n');
load('train.mat'); % Contains train_data [57x2000], train_label [2000x1]
X_train = train_data;
y_train = train_label;

%% Feature Standardization
fprintf('[2/5] Standardizing features (using training stats)...\n');
mean_train = mean(X_train, 2); % mean per row
std_train = std(X_train, 0, 2) + 1e-8; % std per row, add epsilon to avoid division by 0
X_train = (X_train - mean_train) ./ std_train;

%% Quadratic Programming Setup
fprintf('[3/5] Setting up quadratic programming problem...\n');
n_samples = size(X_train, 2);
H = (X_train' * X_train) .* (y_train * y_train'); % H(i,j) = y_i y_j x_i^T x_j
f = -ones(n_samples, 1);

A = [];
b = [];
Aeq = y_train';
Beq = 0;

C = 1e6; % Large C for hard-margin
lb = zeros(n_samples, 1);
ub = ones(n_samples, 1) * C;

%% Solve QP
fprintf('[4/5] Solving quadratic program...\n');
options = optimset('LargeScale', 'off', 'MaxIter', 10000, 'Display', 'off');
Alpha = quadprog(H, f, A, b, Aeq, Beq, lb, ub, [], options);

%% Extract Support Vectors and Parameters
fprintf('[5/5] Computing support vectors and model parameters...\n');
idx = find(Alpha > 1e-4); % threshold to select support vectors
wo = X_train * (Alpha .* y_train); % weight vector w
bo = mean(y_train(idx) - (X_train(:, idx)' * wo)); % bias b

%% Display Results
fprintf('\n=== Linear Kernel SVM Results ===\n');
fprintf('Number of Support Vectors: %d\n', length(idx));
fprintf('w = [%s]\n', num2str(wo'));
fprintf('b = %.4f\n', bo);

%% Evaluate Training Accuracy Only
train_predictions = sign(X_train' * wo + bo);
train_acc = sum(train_predictions == y_train) / length(y_train);
fprintf('Training Accuracy: %.2f%%\n', train_acc * 100);
