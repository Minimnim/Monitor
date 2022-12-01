%gwet's AC1 
function Gwet = gwet(x,y)
    C = confusionmat(x,y); % compute confusion matrix
    n = sum(C(:)); % get total N
    p = (C(1,1)+ C(2,2))/n %overall percent agreement
    A1 = sum(C(:,1))
    B1 = sum(C(1,:))
    q = (A1 + B1)/(2*n)
    e = (2*q)*(1-q) %chance agreement probability
    Gwet = (p-e)/(1-e)
end 
