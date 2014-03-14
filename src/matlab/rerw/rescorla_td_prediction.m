function [x_pre] = rescorla_td_prediction(walkset,choices,learningrate)
% Same as RL_rescorlaTD_miriam.m in pilot 2 with the slight change that now
% not all outcomes are shown and thus the td algorithm only updates the prediction 
% for the one that is shown and leaves the two others the same

% Initialise (step 0 of recursive iteration)
x_pre(:,1)  = [0.5 0.5];  % Initial estimate (mean): no big effect if changed

% Update - for all rewards in all steps because all thre outcomes shown
for t=1:length(walkset)-1
    
    % received reward and prediction error
    x(t)      = walkset(t);
    pe(t)     = x(t) - x_pre(choices(t),t);           
    
    % leave prediction the same for unchosen and update it for chosen stim
    x_pre(:,t+1) = x_pre(:,t);
    x_pre(choices(t),t+1)= (1-learningrate) * x_pre(choices(t),t) + learningrate * x(t);
    % This update rule is the same as "x_pre(:,t) + learningrate * (x(t) -
    % x_pre(:,t)) which is again the same as x_pre(:,t) + learningrate * pe
    
end


