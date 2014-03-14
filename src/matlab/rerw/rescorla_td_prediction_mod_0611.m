function [x_pre] = rescorla_td_prediction_mod_0611(walkset,choices,learningrate,lengthrw1,lengthrw2)
% Same as RL_rescorlaTD_miriam.m in pilot 2 with the slight change that now
% not all outcomes are shown and thus the td algorithm only updates the prediction 
% for the one that is shown and leaves the two others the same

% learn about random walks independent of magnitudes (as they are random)

orig_walkset = walkset;
orig_choices = choices;

% First half
% ===================================================================== %
walkset = orig_walkset(:,1:lengthrw1);  %check this just takes first half
choices = orig_choices(:,1:lengthrw1);

% Init
x_pre1(:,1)  = [0.5 0.5];

% Update - for all rewards in all steps because all thre outcomes shown
for t=1:length(walkset)-1
    
    % received reward and prediction error
    x(t)      = walkset(t);
    pe(t)     = x(t) - x_pre1(choices(t),t);           
    
    % leave prediction the same for unchosen and update it for chosen stim
    x_pre1(:,t+1) = x_pre1(:,t);
    x_pre1(choices(t),t+1)= (1-learningrate) * x_pre1(choices(t),t) + learningrate * x(t);
    % This update rule is the same as "x_pre(:,t) + learningrate * (x(t) -
    % x_pre(:,t)) which is again the same as x_pre(:,t) + learningrate * pe
    
end

clear x t walkset choices pe;

% Second half
% ===================================================================== %
walkset = orig_walkset(:,lengthrw1+1:lengthrw1+lengthrw2);  %check this just takes the second half
choices = orig_choices(:,lengthrw1+1:lengthrw1+lengthrw2);

x_pre2(:,1)  = [0.5 0.5];

% Update - for all rewards in all steps because all thre outcomes shown
for t=1:length(walkset)-1
    
    % received reward and prediction error
    x(t)      = walkset(t);
    pe(t)     = x(t) - x_pre2(choices(t),t);           
    
    % leave prediction the same for unchosen and update it for chosen stim
    x_pre2(:,t+1) = x_pre2(:,t);
    x_pre2(choices(t),t+1)= (1-learningrate) * x_pre2(choices(t),t) + learningrate * x(t);
    % This update rule is the same as "x_pre(:,t) + learningrate * (x(t) -
    % x_pre(:,t)) which is again the same as x_pre(:,t) + learningrate * pe
    
end

x_pre = [x_pre1 x_pre2]; %check it concatenates along right dimensions

