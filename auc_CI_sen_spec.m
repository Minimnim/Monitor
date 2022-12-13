%to upload the file with probabilities and outcome
all = readtable("Monitor_probabilities_AUC.xlsx")
data = table2array(all)
%outcome = readtable("MRI outcome for xgb_58ver.csv")
%outcome = readtable("HIE outcome for xgb_60ver.csv")
outcome = readtable("gray injury outcome for xgb_58ver.csv")
outcome = table2array(outcome)
%mri = outcome(:,2)
%hie = outcome(:,2)
gray = outcome(:,2)
%to have AUC and 95% confidence interval
a = data(1:60,18) %since it has NaN values, before going further, we have to delete them 
a(29,:) = []
a(39,:) = []
[aauc,aauc_ci]=bootstrap_aucs(a,gray)
%to have specificity and sensitivity 
thresh = 0.5
b = zeros(58,1)
for x = 1:58
    if a(x,:) > thresh
        b(x,:) = 1
    else 
        b(x,:) = 0
    end
end 
C = confusionmat(gray,b)
TP = C(2, 2);
TN = C(1, 1);
FP = C(1, 2);
FN = C(2, 1);
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Sensitivity = TP / (FN + TP)
specificity = TN / (TN + FP)
aauc
aauc_ci