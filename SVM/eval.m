function eval_predicted = custom_svm()
    % Load training data
    load('train.mat');
    X_train = train_data; % 57x2000
    y_train = train_label; % 2000x1

    % Load evaluation data
    load('eval.mat');
    X_eval = eval_data; % 57x600

    % Normalize Data (Optional)
    X_train = X_train ./ max(abs(X_train), [], 2);
    X_eval = X_eval ./ max(abs(X_eval), [], 2);

    % Parameters
    C = 1; % Soft-Margin Parameter (Regularization)
    gamma = 0.1; % RBF Kernel Parameter

    %% Compute RBF Kernel Matrix
    fprintf('Computing RBF Kernel Matrix for Training Data...\n');
    K_train = rbfKernel(X_train, X_train, gamma);

    % Formulate Quadratic Programming Problem
    H = (y_train * y_train') .* K_train;
    f = -ones(size(y_train));
    Aeq = y_train';
    Beq = 0;
    lb = zeros(size(y_train));
    ub = ones(size(y_train)) * C;

    %% Solve Using Quadratic Programming
    fprintf('Solving Quadratic Programming Problem using quadprog...\n');
    options = optimset('LargeScale', 'off', 'MaxIter', 50000, 'Display', 'iter');
    Alpha = quadprog(H, f, [], [], Aeq, Beq, lb, ub, [], options);

    %% Calculate Support Vectors and Bias
    fprintf('Calculating Support Vectors and Bias...\n');
    sv_idx = find(Alpha > 1e-4);
    b = mean(y_train(sv_idx) - sum((Alpha .* y_train) .* K_train(:, sv_idx), 1)');

    %% Compute Kernel for Evaluation Data
    fprintf('Computing Kernel for Evaluation Data...\n');
    K_eval = rbfKernel(X_eval, X_train, gamma);

    % Compute Discriminant Function
    eval_predicted = sign(K_eval * (Alpha .* y_train) + b);
    fprintf('Prediction Complete. Evaluation Set Classified.\n');
end

%% RBF Kernel Function
function K = rbfKernel(X1, X2, gamma)
    % Efficient RBF Kernel Calculation using Vectorized Form
    n1 = size(X1, 2);
    n2 = size(X2, 2);
    K = zeros(n1, n2);
    for i = 1:n1
        for j = 1:n2
            diff = X1(:, i) - X2(:, j);
            K(i, j) = exp(-gamma * (diff' * diff));
        end
    end
end
