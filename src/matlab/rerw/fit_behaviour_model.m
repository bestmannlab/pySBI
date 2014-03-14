clear all;

% Load model behavioral data
load ../../../data/rerw/control/choice.csv
load ../../../data/rerw/control/rew.csv
load ../../../data/rerw/control/mag.csv
load ../../../data/rerw/control/prob.csv

% estimate learning rate and beta
nfits=100;
all_param_estimates=zeros(2,nfits);
estimate_energy=zeros(1,nfits);
for i=1:nfits
    params = [rand() rand()];
    options = optimset('MaxFunEvals', 100000,'MaxIter',500000); 
    [all_param_estimates(:,i),estimate_energy(i)]=fminsearch(@energy_learn_rewards_model,params,options,mag,rew,choice);
end
minidx=find(estimate_energy==min(estimate_energy),1);
param_ests=all_param_estimates(:,minidx);

% plot and save 
fit_vals = rescorla_td_prediction(rew,choice,param_ests(1));
fit_probs=zeros(size(fit_vals));
fit_probs(1,:)=1./(1+exp(-param_ests(2)*(fit_vals(1,:)-fit_vals(2,:))));
fit_probs(2,:)=1./(1+exp(-param_ests(2)*(fit_vals(2,:)-fit_vals(1,:))));
prop_correct_vec=zeros(1,100);
for i=1:100
    fit_choices=zeros(size(choice));
    for t=1:length(choice)    
       fit_choices(t)=randsample([1 2],1,true,fit_probs(:,t));       
    end
    prop_correct_vec(i)=length(find(fit_choices-choice==0))/length(choice); 
end
prop_correct=mean(prop_correct_vec);

figure;
subplot(3,1,1); 
title('Real probs');hold on;
plot(prob');

subplot(3,1,2); 
title('Fit vals + rewards');hold on;
plot(fit_vals');
plot(rew,'o');

subplot(3,1,3);
title('Fit sel probs');hold on;
plot(fit_probs');

disp(['Learning rate: ',num2str(param_ests(1))]);
disp(['Beta: ',num2str(param_ests(2))]);
disp(['Proportion of correctly predicted choices: ',num2str(prop_correct)]);
