% test_svm_main.m - Evaluate svm_main.m using labeled test subset with full metrics
clearvars; clc;

% Load training and test data
load('train.mat'); % train_data, train_label
load('test.mat');  % test_data, test_label

% Use 600-sample subset for mock eval
eval_data = test_data(:, 1:600);
eval_label = test_label(1:600);
save('eval.mat', 'eval_data'); % Overwrites eval.mat for svm_main

% Run the SVM model
svm_main;

% Validate eval_predicted
assert(isvector(eval_predicted) && length(eval_predicted) == 600, 'eval_predicted must be 600 x 1');
assert(all(ismember(eval_predicted, [-1, 1])), 'eval_predicted must only contain +1 or -1');

% Accuracy
accuracy = mean(eval_predicted == eval_label) * 100;

% Confusion Matrix
[cm, order] = confusionmat(eval_label, eval_predicted);
TP = cm(2,2);
TN = cm(1,1);
FP = cm(1,2);
FN = cm(2,1);

% Metrics
precision = TP / (TP + FP);
recall    = TP / (TP + FN);
f1_score  = 2 * (precision * recall) / (precision + recall);

% Display Results
fprintf('\n--- Classification Metrics ---\n');
fprintf('Accuracy: %.2f%%\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1_score);

fprintf('\nConfusion Matrix:\n');
fprintf('True Positive: %d\n', TP);
fprintf('True Negative: %d\n', TN);
fprintf('False Positive: %d\n', FP);
fprintf('False Negative: %d\n', FN);
