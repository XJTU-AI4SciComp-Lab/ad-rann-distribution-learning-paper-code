function T = black_scholes_training_history_table(history)
%BLACK_SCHOLES_TRAINING_HISTORY_TABLE Convert stored Adam states to table.

    iteration = history.iteration(:);
    r_s = history.p(1,:).';
    r_t = history.p(2,:).';
    objective = history.objective(:);
    residual_mse = history.residual_mse(:);
    gradient_norm = history.grad_norm(:);
    step_norm = history.step_norm(:);
    evaluation_time_sec = history.eval_time(:);

    selected = false(numel(iteration),1);
    selected(history.best_index) = true;

    T = table( ...
        iteration,r_s,r_t,objective,residual_mse,gradient_norm, ...
        step_norm,evaluation_time_sec,selected);
end
