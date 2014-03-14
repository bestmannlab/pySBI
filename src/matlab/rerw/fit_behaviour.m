clear all;

%load ../../../data/rerw/subjects/1_2.mat
load ../../../data/rerw/subjects/value1_s3_t2.mat

rh_button = 7;
lh_button = 22;
choice(store.dat.RESP==rh_button)=1;
choice(store.dat.RESP==lh_button)=2;

probs = store.dat.probswalk;
mags = store.dat.mags;
rew = store.dat.outcrec;

% take care of kicking out premature/missed/late responses etc
missed = find(isnan(store.dat.RESP));
probs(:,missed)=[];
mags(:,missed)=[];
rew(missed) = [];
choice(missed)=[];

% trials = [101:length(probs)];%[1:100];%
% probs = probs(:,trials);
% mags = mags(:,trials);
% rew = rew(trials);
% choice = choice(trials);

mags = mags/100;

% initialise values - try different ones, or grid appraoch etc.
alpha = 0.5; 
beta =  4;   

% estimate learning rate and beta
nfits=100;
all_param_estimates=zeros(2,nfits);
estimate_energy=zeros(1,nfits);
for i=1:nfits
    params = [rand() rand()];
    options = optimset('MaxFunEvals', 100000,'MaxIter',500000); 
    [all_param_estimates(:,i),estimate_energy(i)]=fminsearch(@energy_learn_rewards,params,options,mags,rew,choice);
end
minidx=find(estimate_energy==min(estimate_energy),1);
param_ests=all_param_estimates(:,minidx);

% plot and save 
%model_vals = rescorla_td_prediction(rew,choice,param_ests(1));
%model_probs=1./(1+exp(-param_ests(2)*model_vals.*mags));
%[model_max_prob,model_choices]=max(model_probs);
%prop_correct=length(find(model_choices-choice==0))/length(choice);
fit_vals = rescorla_td_prediction(rew,choice,param_ests(1));
fit_probs=zeros(size(fit_vals));
ev=fit_vals.*mags;
fit_probs(1,:)=1./(1+exp(-param_ests(2)*(ev(1,:)-ev(2,:))));
fit_probs(2,:)=1./(1+exp(-param_ests(2)*(ev(2,:)-ev(1,:))));
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
subplot(3,1,1); hold on; title('Real probs');
plot(probs');

subplot(3,1,2); hold on; title('Modelled probs + rewards');
plot(fit_probs');hold on;
plot(rew,'o');

subplot(3,1,3); hold on; title('Modelled vals - chosen vs unchosen');
for t=1:length(probs)
    ch_val(t) = fit_vals(choice(t),t);
    unch_val(t) = fit_vals(3-choice(t),t);
end
plot(ch_val);hold on;plot(unch_val,'r');

disp(['Learning rate: ',num2str(param_ests(1))]);
disp(['Beta: ',num2str(param_ests(2))]);
disp(['Proportion of correctly predicted choices: ',num2str(prop_correct)]);
