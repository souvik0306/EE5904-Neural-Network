% test_svm_main.m - Validate svm_main.m using part of test.mat
clearvars; clc;

% Load train and test data
load('train.mat'); % train_data, train_label
load('test.mat');  % test_data, test_label

% Assign to variables
X_train = train_data;
y_train = train_label;

% Use first 300 test samples as mock eval set
eval_data = test_data(:, 1:300);
eval_label = test_label(1:300);
save('eval.mat', 'eval_data'); % Write mock eval.mat file

% Run Task 3 model
svm_main;

% Evaluate
assert(isvector(eval_predicted) && length(eval_predicted) == 300, 'eval_predicted must be 300 x 1');
assert(all(ismember(eval_predicted, [-1, 1])), 'Output must contain only +1 or -1');

% Accuracy
acc = mean(eval_predicted == eval_label) * 100;
fprintf('Test Accuracy on 300-sample subset: %.2f%%\n', acc);

% Confusion matrix
disp('Confusion Matrix:');
disp(confusionmat(eval_label, eval_predicted));
