clear all;

% Load reward probabilities and magnitudes from data file
load ../../../data/rerw/subjects/1_2.mat
probs = store.dat.probswalk;
mags = store.dat.mags;
mags = mags/100;

% Model parameters
alpha = 0.2; 
beta =  10;

% Simulate Rescorla-Wagner rule with softmax using these parameters
% Choices made by the model on each trial
model_choices=zeros(1,length(probs));
% Rewards received by the model on each trial
model_rewards=zeros(1,length(probs));
% Values (expected reward) estimated for each option by the model on each trial
model_vals=zeros(2,length(probs));
% Choice selection probabilities computed by the model on each trial
model_probs=zeros(2,length(probs));

% Initialize expected reward
exp_rew=[0.5 0.5];

for i=1:length(probs)
    % Log expected reward
    model_vals(:,i)=exp_rew;
    % Compute softmax on expected reward
    ev=model_vals(:,i).*mags(:,i);
    model_probs(1,i)=1./(1+exp(-beta*(ev(1,:)-ev(2,:))));
    model_probs(2,i)=1./(1+exp(-beta*(ev(2,:)-ev(1,:))));
    % Make choice
    model_choices(i)=randsample([1 2],1,true,model_probs(:,i));
    % Receive reward
    if rand()<=probs(model_choices(i),i)
        model_rewards(i)=1.0;
    end
    % Update expected reward - Rescorla-Wagner rule
    exp_rew(model_choices(i))=(1-alpha)*exp_rew(model_choices(i))+alpha*model_rewards(i);
end

% estimate learning rate and beta
% Initialize parameter guesses to random values
nfits=100;
all_param_estimates=zeros(2,nfits);
estimate_energy=zeros(1,nfits);
for i=1:nfits
    params = [rand() rand()];
    options = optimset('MaxFunEvals', 100000,'MaxIter',500000); 
    [all_param_estimates(:,i),estimate_energy(i)]=fminsearch(@energy_learn_rewards,params,options,mags,model_rewards,model_choices);
end
minidx=find(estimate_energy==min(estimate_energy),1);
param_ests=all_param_estimates(:,minidx);

% plot and save 
fit_vals = rescorla_td_prediction(model_rewards,model_choices,param_ests(1));
fit_probs=zeros(size(fit_vals));
ev=fit_vals.*mags;
fit_probs(1,:)=1./(1+exp(-param_ests(2)*(ev(1,:)-ev(2,:))));
fit_probs(2,:)=1./(1+exp(-param_ests(2)*(ev(2,:)-ev(1,:))));
prop_correct_vec=zeros(1,100);
for i=1:100
    fit_choices=zeros(size(model_choices));
    for t=1:length(model_choices)    
       fit_choices(t)=randsample([1 2],1,true,fit_probs(:,t));       
    end
    prop_correct_vec(i)=length(find(fit_choices-model_choices==0))/length(model_choices); 
end
prop_correct=mean(prop_correct_vec);

figure;
subplot(5,1,1); 
title('Real probs');hold on;
plot(probs');

subplot(5,1,2); 
title('Model vals + rewards');hold on;
plot(model_vals');
plot(model_rewards,'o');

subplot(5,1,3); 
title('Fit vals + rewards');hold on;
plot(fit_vals');
plot(model_rewards,'o');

subplot(5,1,4);
title('Model sel probs');hold on;
plot(model_probs');

subplot(5,1,5);
title('Fit sel probs');hold on;
plot(fit_probs');

disp(['Learning rate: ',num2str(param_ests(1))]);
disp(['Beta: ',num2str(param_ests(2))]);
disp(['Proportion of correctly predicted choices: ',num2str(prop_correct)]);
