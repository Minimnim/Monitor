P = 'C:\Users\PhysioUser\Desktop\PhD\Monitor\Minoo\pre-processed signals\NIRS containing transients\mydata';
S = dir(fullfile(P,'*.txt')); 
params = decomp_PARAMS
fs = 1 / 6;
db_plot = false;
for k = 30 %1:numel(S)
    F = fullfile(P,S(k).name);
    S(k).data = readtable(F);
    data = table2array(S(k).data)
    sat = data(:,2)
    sat = fillmissing(sat, 'linear')
    y = shorttime_iter_SSA_decomp(sat, fs, params, db_plot);
    transient = y.component.'
    nirs = y.nirs.'
    without_component = nirs - transient + nanmean(sat)
    tran = append('transient', '_' ,  erase(S(k).name, 'mydata_'))
    fil = append('filtered', '_' ,  erase(S(k).name, 'mydata_'))
    csvwrite(tran,transient)
    csvwrite(fil,without_component)
end 
