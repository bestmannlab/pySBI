function energy = energy_learn_rewards(params,mags,rewards,choice)
% First I need to use the current parameters to calculate the expected
% rewards using the Rescorla-Wagner, then I can calculate the
% probability of choosing each of the three parameters based on those
% expected outcomes, and from that the loglikelihood that those parameters
% explained the actual choices the subject made.
%
% Params contains the parameters that are to be estimated which are:
% alpha               between 0 and 1; learning rate
% beta                exploration parameter going into the softmax function
%                     to obtain the probability of choice of each stimulus
%                     given the predictions
%                     i.e. params = [6 3 0.9 3];
%
% This function gives back an energy function which will be minimised in
% fit_behaviour using fminsearch


% set parameters
alpha   = params(1);    % learning rate
beta = params(2);


% don't allow negative estimates
if(any(params<=0)) | alpha>1
    energy=10000000; return;
end

% get modelled probs using Rescorla-Wagner model, then vals, then energy
model_probs = rescorla_td_prediction(rewards,choice,alpha);
model_vals = model_probs.*mags;
ch_valdiffs=model_vals(1,:)-model_vals(2,:);
ch_valdiffs(find(choice==2))=-ch_valdiffs(find(choice==2)); % swap if RH chosen
energy=-sum(log(1./(1+exp(-beta*ch_valdiffs)))); % maximise log likelihood (softmax function)
