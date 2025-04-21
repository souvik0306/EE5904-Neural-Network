% Task 3: SVM with Gaussian (RBF) Kernel and Class Weighting for Best Performance
rng(42); % Reproducibility

%% Load Data
load('train.mat'); % train_data [57x2000], train_label [2000x1]
load('eval.mat');  % eval_data [57x600]
X_train = train_data';
y_train = train_label;
X_eval = eval_data';

%% Normalize Features
mu = mean(X_train);
sigma = std(X_train) + 1e-8;
X_train = (X_train - mu) ./ sigma;
X_eval  = (X_eval  - mu) ./ sigma;

%% Define Grid for Hyperparameter Search
C_values = [0.1, 1, 10, 100];
gamma_values = [0.001, 0.01, 0.1, 1];
best_f1 = -inf;

for C = C_values
    for gamma = gamma_values
        f1_scores = [];
        cv = cvpartition(y_train, 'KFold', 5);

        for i = 1:cv.NumTestSets
            idx_train = training(cv, i);
            idx_val = test(cv, i);

            model = fitcsvm(X_train(idx_train, :), y_train(idx_train), ...
                'KernelFunction', 'rbf', ...
                'KernelScale', 1/sqrt(2*gamma), ...
                'BoxConstraint', C, ...
                'Standardize', false, ...
                'ClassNames', [-1, 1], ...
                'Weights', get_class_weights(y_train(idx_train)));

            [pred, score] = predict(model, X_train(idx_val, :));
            [~, ~, ~, f1] = compute_metrics(y_train(idx_val), pred);
            f1_scores(end+1) = f1;
        end

        mean_f1 = mean(f1_scores);
        if mean_f1 > best_f1
            best_f1 = mean_f1;
            best_model = model;
            best_C = C;
            best_gamma = gamma;
        end
    end
end

fprintf('Best model: C = %.2f, gamma = %.3f, F1 = %.4f\n', best_C, best_gamma, best_f1);

%% Retrain Best Model on Full Data
final_model = fitcsvm(X_train, y_train, ...
    'KernelFunction', 'rbf', ...
    'KernelScale', 1/sqrt(2*best_gamma), ...
    'BoxConstraint', best_C, ...
    'Standardize', false, ...
    'ClassNames', [-1, 1], ...
    'Weights', get_class_weights(y_train));

%% Predict on Eval
eval_predicted = predict(final_model, X_eval);
eval_predicted = eval_predicted(:); % ensure column vector

fprintf('Evaluation complete. Output: eval_predicted [%d x 1]\n', length(eval_predicted));

%% --- Helper Functions ---

function W = get_class_weights(y)
    % Balanced class weights to improve recall
    n_pos = sum(y == 1);
    n_neg = sum(y == -1);
    W = ones(size(y));
    W(y == 1) = 0.5 / n_pos;
    W(y == -1) = 0.5 / n_neg;
end

function [acc, prec, rec, f1] = compute_metrics(y_true, y_pred)
    tp = sum(y_true == 1 & y_pred == 1);
    tn = sum(y_true == -1 & y_pred == -1);
    fp = sum(y_true == -1 & y_pred == 1);
    fn = sum(y_true == 1 & y_pred == -1);

    acc = (tp + tn) / length(y_true);
    prec = tp / (tp + fp + eps);
    rec = tp / (tp + fn + eps);
    f1 = 2 * prec * rec / (prec + rec + eps);
end
