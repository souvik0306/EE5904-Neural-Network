
% Load reward matrix
load('task1.mat');

% Reproducibility
seed = 42;
rng(seed);

% Settings
decay_types = 4;
gammas = [0.5, 0.9];
max_trials = 3000;
conv_threshold = 0.05;
runs_per_setting = 10;

% Metrics
best_overall_reward = -Inf;
best_overall_path = [];
best_overall_Q = [];
best_overall_title = "";
rewards = NaN(decay_types, 2);  % best reward per ε-type × γ
epsilon_logs = cell(decay_types, 1);  % for εₖ comparison graph

% Main loop
for eps_type = 1:decay_types
    for g_idx = 1:length(gammas)
        gamma = gammas(g_idx);
        fprintf('\n[INFO] γ = %.1f | ε-type = %d\n', gamma, eps_type);

        goal_reaches = 0;
        total_time = 0;
        max_reward = -Inf;
        best_path = [];
        best_Q = [];

        for r = 1:runs_per_setting
            fprintf('  → Run %2d/%d\n', r, runs_per_setting);
            rng(seed + r);
            Q = zeros(size(reward));
            tic;

            for trial = 1:max_trials
                Q_prev = Q;
                s_k = 1; k = 1;
                eps_trace = [];  % store εₖ for this trial

                while s_k ~= 100
                    eps_k = getEps(k, eps_type);
                    if eps_k < 0.005, break; end
                    eps_trace(end+1) = eps_k;

                    a_k = nextAct(Q(s_k,:), eps_k, reward(s_k,:));
                    delta = 10^(mod(a_k + 1, 2)) * (-1)^(floor(a_k / 2) + 1);
                    s_k_next = s_k + delta;

                    Q(s_k, a_k) = Q(s_k, a_k) + eps_k * ...
                        (reward(s_k, a_k) + gamma * max(Q(s_k_next, :)) - Q(s_k, a_k));

                    s_k = s_k_next;
                    k = k + 1;
                end

                if max(abs(Q - Q_prev), [], 'all') < conv_threshold
                    break;
                end
            end

            [total_reward, path] = evaluatePolicy(Q, gamma, reward);
            if ~isempty(path) && path(end) == 100
                goal_reaches = goal_reaches + 1;
                total_time = total_time + toc;
                if total_reward > max_reward
                    max_reward = total_reward;
                    best_path = path;
                    best_Q = Q;

                    % Save one representative epsilon trace
                    if isempty(epsilon_logs{eps_type})
                        epsilon_logs{eps_type} = eps_trace;
                    end
                end
            end
        end

        rewards(eps_type, g_idx) = max_reward;

        if max_reward > best_overall_reward
            best_overall_reward = max_reward;
            best_overall_path = best_path;
            best_overall_Q = best_Q;
            best_overall_title = sprintf('γ = %.1f | ε-type = %d | Reward = %.2f', gamma, eps_type, max_reward);
        end

        fprintf('[RESULT] Reached goal in %d/%d runs | Best reward = %.2f\n', ...
                goal_reaches, runs_per_setting, max_reward);
    end
end

% === Plot: Optimal policy
plotPolicy(best_overall_Q, best_overall_title);

% === Plot: Execution of best path
plotPath(best_overall_path, best_overall_title);

% === Plot: Epsilon decay comparison (from actual RL runs)
figure;
hold on;
colors = lines(decay_types);
labels = {'1/k', '100 / (100 + k)', '(1 + log(k)) / k', '(1 + 5 log(k)) / k'};
for i = 1:decay_types
    if ~isempty(epsilon_logs{i})
        plot(1:length(epsilon_logs{i}), epsilon_logs{i}, 'LineWidth', 2, 'Color', colors(i,:));
    end
end
hold off;
legend(labels, 'Location', 'northeast');
xlabel('Step k'); ylabel('\epsilon_k');
title('Comparison of \epsilon_k Decay During Q-Learning');
grid on;

%% === Helper Functions ===

function eps_k = getEps(k, mode)
    switch mode
        case 1, eps_k = 1 / k;
        case 2, eps_k = 100 / (100 + k);
        case 3, eps_k = (1 + log(k)) / k;
        case 4, eps_k = (1 + 5 * log(k)) / k;
        otherwise, error('Invalid epsilon decay type');
    end
end

function a_k = nextAct(Q_s, eps_k, reward_s)
    valid = find(reward_s ~= -1);
    if any(Q_s)
        if rand > eps_k
            [~, idx] = max(Q_s(valid));
            a_k = valid(idx);
        else
            rand_idx = find(Q_s(valid) ~= max(Q_s(valid)));
            if isempty(rand_idx)
                a_k = valid(randi(length(valid)));
            else
                a_k = valid(rand_idx(randi(length(rand_idx))));
            end
        end
    else
        a_k = valid(randi(length(valid)));
    end
end

function [total_reward, path] = evaluatePolicy(Q, gamma, reward)
    [~, actions] = max(Q, [], 2);
    s = 1; path = []; total_reward = 0; discount = 1;
    visited = zeros(1, 100);
    while s ~= 100 && visited(s) == 0
        path(end+1) = actions(s);
        visited(s) = 1;
        total_reward = total_reward + discount * reward(s, actions(s));
        s = s + (10 ^ (mod(actions(s) + 1, 2)) * (-1) ^ (floor(actions(s) / 2) + 1));
        discount = discount * gamma;
    end
    if s == 100
        path(end+1) = 100;
    else
        path = [];
    end
end

function plotPolicy(Q, title_str)
    [~, actions] = max(Q, [], 2);
    dirs = ['^r'; '>r'; 'vr'; '<r'];
    figure; hold on;
    axis([0 10 0 10]); grid on;
    title(['Optimal Policy | ', title_str]);
    set(gca,'YDir','reverse');
    plot(0.5, 0.5, '*g'); plot(9.5, 9.5, '*r');

    for i = 1:100
        x = floor((i - 1) / 10) + 0.5;
        y = mod(i - 1, 10) + 0.5;
        plot(x, y, dirs(actions(i), :));
    end
    hold off;
end

function plotPath(path, title_str)
    dirs = ['^b'; '>b'; 'vb'; '<b'];
    figure; hold on;
    axis([0 10 0 10]); grid on;
    title(['Execution of Optimal Path | ', title_str]);
    set(gca,'YDir','reverse');
    plot(0.5, 0.5, '*g'); plot(9.5, 9.5, '*r');
    pos = 1;
    for i = 1:length(path)-1
        x = floor((pos - 1) / 10) + 0.5;
        y = mod(pos - 1, 10) + 0.5;
        a_k = path(i);
        plot(x, y, dirs(a_k, :), 'LineWidth', 2);
        delta = 10 ^ (mod(a_k + 1, 2)) * (-1) ^ (floor(a_k / 2) + 1);
        pos = pos + delta;
    end
    hold off;
end
