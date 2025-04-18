clc; clear; close all;

% -------------------------------
% Load reward matrix from Task 2
% -------------------------------
load('qeval.mat');  % contains qevalreward
reward = qevalreward;
[num_state, num_action] = size(reward);

% -------------------------------
% Parameters
% -------------------------------
gamma = 0.95;
rate_mode = 2;  % Choose from 1 to 5
max_trials = 3000;
convergence_threshold = 0.005;
num_runs = 10;

% -------------------------------
% Track best result
% -------------------------------
reach_count = 0;
max_reward = -Inf;
run_times = [];
best_info = [];

% -------------------------------
% Run multiple trials
% -------------------------------
for run = 1:num_runs
    fprintf('Run %d/%d\n', run, num_runs);
    Q = zeros(num_state, num_action);
    trial = 1;
    start_state = 1;
    end_state = 100;
    converged = false;
    tic;

    while trial <= max_trials && ~converged
        k = 1; s_k = start_state;
        Q_prev = Q;

        while s_k ~= end_state
            [~, eps_k] = selectRate(k, rate_mode);
            if eps_k < 0.005, break; end

            a_k = selectAction(Q(s_k,:), eps_k, reward(s_k,:));
            [Q(s_k,a_k), s_k] = updateQ(Q, reward, s_k, a_k, eps_k, gamma);
            k = k + 1;
        end

        trial = trial + 1;
        converged = checkConvergence(Q_prev, Q, convergence_threshold);
    end

    elapsed = toc;
    [success, policy, path_reward, qevalstates, arrows_x, arrows_y, labels, total_reward] = ...
        extractPolicy(Q, reward, start_state, end_state, gamma);

    if success
        reach_count = reach_count + 1;
        run_times(end+1) = elapsed;
        if total_reward > max_reward
            max_reward = total_reward;
            best_info = {arrows_x, arrows_y, labels, qevalstates, total_reward};
        end
    end
end

% -------------------------------
% Output Best Result
% -------------------------------
fprintf('\nMax reward achieved: %.2f\n', max_reward);
fprintf('Reached goal in %d/%d runs\n', reach_count, num_runs);
fprintf('Average time per successful run: %.3fs\n', mean(run_times));
qevalstates = best_info{4};
disp('States visited:'); disp(qevalstates);

% -------------------------------
% Plot trajectory
% -------------------------------
figure;
title(sprintf('Execution on qeval.mat | Reward = %.2f', best_info{5}));
set(gca, 'YDir', 'reverse'); axis([0 10 0 10]); grid on; hold on;
xlabel('Column'); ylabel('Row');
plot(0.5, 0.5, '*g'); plot(9.5, 9.5, '*r');  % Start and Goal
for i = 1:length(best_info{1})
    scatter(best_info{1}(i), best_info{2}(i), 75, best_info{3}(i*2-1), best_info{3}(i*2));
end
hold off;

% -------------------------------
% Helper Functions
% -------------------------------
function [title_str, rate] = selectRate(k, mode)
    switch mode
        case 1, rate = 1 / k;                      title_str = '1/k';
        case 2, rate = 100 / (100 + k);            title_str = '100 / (100 + k)';
        case 3, rate = (1 + log(k)) / k;           title_str = '(1 + log(k)) / k';
        case 4, rate = (1 + 5 * log(k)) / k;       title_str = '(1 + 5log(k)) / k';
        case 5, rate = exp(-0.001 * k);            title_str = 'exp(-0.001k)';
        otherwise, error('Invalid rate mode.');
    end
end

function a = selectAction(Q_s, eps_k, reward_s)
    valid = find(reward_s ~= -1);
    if any(Q_s)
        if rand > eps_k
            [~, idx] = max(Q_s(valid));
            a = valid(idx);
        else
            rand_idx = find(Q_s(valid) ~= max(Q_s(valid)));
            if isempty(rand_idx)
                a = valid(randi(length(valid)));
            else
                a = valid(rand_idx(randi(length(rand_idx))));
            end
        end
    else
        a = valid(randi(length(valid)));
    end
end

function [Q_sa, s_next] = updateQ(Q, reward, s, a, lr, gamma)
    switch a
        case 1, s_next = s - 1;
        case 2, s_next = s + 10;
        case 3, s_next = s + 1;
        case 4, s_next = s - 10;
    end
    Q_sa = Q(s, a) + lr * (reward(s, a) + gamma * max(Q(s_next, :)) - Q(s, a));
end

function flag = checkConvergence(Q_old, Q_new, tol)
    flag = max(abs(Q_old - Q_new), [], 'all') < tol;
    if flag, disp("Converged."); end
end

function [success, policy, policy_reward, state_list, x, y, label, total_reward] = ...
    extractPolicy(Q, reward, s_state, e_state, gamma)
    [~, policy] = max(Q, [], 2);
    s = s_state;
    step = 1;
    total_reward = 0;
    x = []; y = []; label = [];
    policy_reward = zeros(10, 10);
    state_list = [];

    while s ~= e_state && policy_reward(mod(s,10),floor((s-1)/10)+1) == 0
        state_list = [state_list, s];
        x = [x, floor((s-1)/10) + 0.5];
        y = [y, mod(s-1,10) + 0.5];

        switch policy(s)
            case 1, label = [label, '^b']; policy_reward(mod(s,10),floor((s-1)/10)+1) = gamma.^(step-1)*reward(s,1); s = s - 1;
            case 2, label = [label, '>b']; policy_reward(mod(s,10),floor((s-1)/10)+1) = gamma.^(step-1)*reward(s,2); s = s + 10;
            case 3, label = [label, 'vb']; policy_reward(mod(s,10),floor((s-1)/10)+1) = gamma.^(step-1)*reward(s,3); s = s + 1;
            case 4, label = [label, '<b']; policy_reward(mod(s,10),floor((s-1)/10)+1) = gamma.^(step-1)*reward(s,4); s = s - 10;
        end
        step = step + 1;
    end

    if s == e_state
        state_list = [state_list, s];
        x = [x, 9.5];
        y = [y, 9.5];
        label = [label, 'pr'];
        success = true;
        disp("Reached end state.");
    else
        success = false;
        disp("Did not reach end state.");
    end

    total_reward = sum(policy_reward, 'all');
end
