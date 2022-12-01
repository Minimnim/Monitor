% this code puts together all the features in the single csv file, called
% "all.csv" 
P = 'C:\Users\PhysioUser\Desktop\PhD\Monitor\Minoo\pre-processed signals\Transients\transient_features';
S = dir(fullfile(P,'*.txt')); 
all = zeros(65,15)
for k = 1:numel(S)
    F = fullfile(P,S(k).name);
    S(k).data = readtable(F)
    data = table2array(S(k).data)
    all(k,:) = data
end
csvwrite('all.csv',all)