%Cohen's Kappa
function kappa = Kappa(x,y)
C = confusionmat(x,y); % compute confusion matrix
n = sum(C(:)); % get total N
p = (C(1,1)+C(2,2))/n %overall percent agreement
A1 = sum(C(:,1))
A2 = sum(C(:,2))
B1 = sum(C(1,:))
B2 = sum(C(2,:))
e = ((A1/n)*(B1/n)) + ((A2/n)*(B2/n)) %chance agreement probability 
kappa = (p-e)/(1-e)
end 
