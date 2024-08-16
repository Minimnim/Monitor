%P = 'C:\Users\PhysioUser\Desktop\PhD\Monitor\Minoo\pre-processed signals\Filtered signals';
%S = dir(fullfile(P,'*.txt')); 
PP = 'C:\Users\PhysioUser\Desktop\PhD\Monitor\Minoo\pre-processed signals\NIRS containing transients\mydata';
SS = dir(fullfile(PP,'*.txt')); 
for k = 1:numel(SS)
    clear table
    clear mydata
    %F = fullfile(P,S(k).name);
    %S(k).data = readtable(F);
    %filtered = table2array(S(k).data)
    FF = fullfile(PP,SS(k).name);
    SS(k).data = readtable(FF);
    mydata = table2array(SS(k).data)
    ts = mydata(:,1)
    %to find the energy of the 5th and 95th datapoint of the signal
    d = length(ts)/20
    fifth = round(d)
    nintyfifth = length(ts) - fifth
    nirs = mydata(:,2)
    nirs = fillmissing(nirs, 'pchip')
    nirs_fifth = nirs(fifth, :)
    nirs_nintyfifth = nirs(nintyfifth, :)
    energy5 = (abs(hilbert(nirs_fifth))).^2
    energy95 = (abs(hilbert(nirs_nintyfifth))).^2
    %to decompose the signal into 5 bandwidths 
    [bp1, bp2, bp3, bp4, bp5] = bpfiltn(nirs)
%to decompose each bandwidth into epochs of 4 hours with 2 hours overlap;
%if we want to have have epochs of 12 hours with 6 hours of overlap,
%instead of 1200 we should have 3600, and instead of 2400, we have 7200
    a = length(nirs)/1200
    b = floor(a)
    c = a - floor(a)
    time_epoch = zeros(2400,b)
    time_epoch(:,1) = ts(1:2400,:)
    for x = 2:b-1
        time_epoch(:,x) = ts(((x-1)*1200)+1:(x+1)*1200,:)
    end
    %time_epoch(:,b) = ts(((b-1)*1200)+1:(b+1)*1200,:)
    bp1_epoch = zeros(2400,b);
    bp1_epoch(:,1) = bp1(1:2400,:)
    for x = 2:b-1
        bp1_epoch(:,x) = bp1(((x-1)*1200)+1:(x+1)*1200,:)
    end
    bp2_epoch = zeros(2400,b);
    bp2_epoch(:,1) = bp2(1:2400,:)
    for x = 2:b-1
        bp2_epoch(:,x) = bp2(((x-1)*1200)+1:(x+1)*1200,:)
    end
    bp3_epoch = zeros(2400,b);
    bp3_epoch(:,1) = bp3(1:2400,:)
    for x = 2:b-1
        bp3_epoch(:,x) = bp3(((x-1)*1200)+1:(x+1)*1200,:)
    end   
    bp4_epoch = zeros(2400,b);
    bp4_epoch(:,1) = bp4(1:2400,:)
    for x = 2:b-1
        bp4_epoch(:,x) = bp4(((x-1)*1200)+1:(x+1)*1200,:)
    end     
    bp5_epoch = zeros(2400,b);
    bp5_epoch(:,1) = bp5(1:2400,:)
    for x = 2:b-1
        bp5_epoch(:,x) = bp5(((x-1)*1200)+1:(x+1)*1200,:)
    end  
%to extract the envelop of each epoch in each bandwidth
    env1 = (abs(hilbert(bp1_epoch))).^2
    env2 = (abs(hilbert(bp2_epoch))).^2
    env3 = (abs(hilbert(bp3_epoch))).^2
    env4 = (abs(hilbert(bp4_epoch))).^2
    env5 = (abs(hilbert(bp5_epoch))).^2
%to extract the intantanous frequency of each epoch in each bandwidth
    infreq_bp1 = zeros(2399,b)
    for x = 1:b-1
        infreq_bp1(:,x) = instfreq(bp1_epoch(:,x), time_epoch(:,x), 'Method', 'hilbert')
    end 
   
    infreq_bp2 = zeros(2399,b)
    for x = 1:b-1
        infreq_bp2(:,x) = instfreq(bp2_epoch(:,x), time_epoch(:,x), 'Method', 'hilbert')
    end 

    infreq_bp3 = zeros(2399,b)
    for x = 1:b-1
        infreq_bp3(:,x) = instfreq(bp3_epoch(:,x), time_epoch(:,x), 'Method', 'hilbert')
    end   
    infreq_bp4 = zeros(2399,b)
    for x = 1:b-1
        infreq_bp4(:,x) = instfreq(bp4_epoch(:,x), time_epoch(:,x), 'Method', 'hilbert')
    end 
    infreq_bp5 = zeros(2399,b)
    for x = 1:b-1
        infreq_bp5(:,x) = instfreq(bp3_epoch(:,x), time_epoch(:,x), 'Method', 'hilbert')
    end 
    %to extract the fractal dimension of each epoch in each bandwidth
    fd1 = zeros(1,b)
    for x = 1:b
        fd1(1,x) = fd_higuchi(bp1_epoch(:,x))
    end 
    fd2 = zeros(1,b)
    for x = 1:b
        fd2(1,x) = fd_higuchi(bp2_epoch(:,x))
    end 
    fd3 = zeros(1,b)
    for x = 1:b
        fd3(1,x) = fd_higuchi(bp3_epoch(:,x))
    end 
    fd4 = zeros(1,b)
    for x = 1:b
        fd4(1,x) = fd_higuchi(bp4_epoch(:,x))
    end 
    fd5 = zeros(1,b)
    for x = 1:b
        fd5(1,x) = fd_higuchi(bp5_epoch(:,x))
    end 
    %to summarize the extracted features using mean, std, kurtosis, skewness in
    %one table
    u = repelem(energy5,b)
    i = repelem(energy95,b)
    table = table(mean(env1).', std(env1).', kurtosis(bp1_epoch).', skewness(bp1_epoch).', prctile(env1,5).', prctile(env1,95).', nanmean(infreq_bp1).',nanstd(infreq_bp1).', kurtosis(infreq_bp1).', skewness(infreq_bp1).', prctile(infreq_bp1,5).', prctile(infreq_bp1,95).', fd1.', mean(env2).', std(env2).', kurtosis(bp2_epoch).', skewness(bp2_epoch).', prctile(env2,5).', prctile(env2,95).',nanmean(infreq_bp2).',nanstd(infreq_bp2).', kurtosis(infreq_bp2).', skewness(infreq_bp2).',prctile(infreq_bp2,5).', prctile(infreq_bp2,95).', fd2.', mean(env3).', std(env3).',kurtosis(bp3_epoch).', skewness(bp3_epoch).',  prctile(env3,5).', prctile(env3,95).',nanmean(infreq_bp3).',nanstd(infreq_bp3).', kurtosis(infreq_bp3).', skewness(infreq_bp3).', prctile(infreq_bp3,5).', prctile(infreq_bp3,95).', fd3.', mean(env4).', std(env4).', kurtosis(bp4_epoch).', skewness(bp4_epoch).',  prctile(env4,5).', prctile(env4,95).',nanmean(infreq_bp4).',nanstd(infreq_bp4).', kurtosis(infreq_bp4).', skewness(infreq_bp4).', prctile(infreq_bp4,5).', prctile(infreq_bp4,95).', fd4.', mean(env5).', std(env5).', kurtosis(bp5_epoch).', skewness(bp5_epoch).',  prctile(env5,5).', prctile(env5,95).', nanmean(infreq_bp5).',nanstd(infreq_bp5).', kurtosis(infreq_bp5).', skewness(infreq_bp5).', prctile(infreq_bp5,5).', prctile(infreq_bp5,95).', fd5.', u.', i.')
    %to test if the remainder of the dividing signal in epochs is less than
    %half of the epoch or not. In this case, if the remainder was less than 2
    %hours, we discarded it, if it was more than 2 hours, the features were
    %extarcted from it. if the epochs of 12 hours, instead of 1200 we should
    %write 3600. 
    if c > 0.5
       e =  ts(((b-1)*1200)+1:length(ts),:)
       bp1_epoch_extra = bp1(((b-1)*1200)+1:length(ts),:)
       bp2_epoch_extra = bp2(((b-1)*1200)+1:length(ts),:)
       bp3_epoch_extra = bp3(((b-1)*1200)+1:length(ts),:)
       bp4_epoch_extra = bp4(((b-1)*1200)+1:length(ts),:)
       bp5_epoch_extra = bp5(((b-1)*1200)+1:length(ts),:)
       infreq_bp1_extra = instfreq(bp1_epoch_extra, e, 'Method', 'hilbert')
       infreq_bp2_extra = instfreq(bp2_epoch_extra, e, 'Method', 'hilbert')
       infreq_bp3_extra = instfreq(bp3_epoch_extra, e, 'Method', 'hilbert')
       infreq_bp4_extra = instfreq(bp4_epoch_extra, e, 'Method', 'hilbert')
       infreq_bp5_extra = instfreq(bp5_epoch_extra, e, 'Method', 'hilbert')
       env1_extra = (abs(hilbert(bp1_epoch_extra))).^2
       table(b,1)= array2table(mean(env1_extra))
       table(b,2)= array2table(std(env1_extra))
       table(b,3)= array2table(kurtosis(bp1_epoch_extra))
       table(b,4)= array2table(skewness(bp1_epoch_extra))
       table(b,5)= array2table(prctile(env1_extra,5))
       table(b,6)= array2table(prctile(env1_extra,95))
       table(b,7)= array2table(nanmean(infreq_bp1_extra))
       table(b,8)= array2table(nanstd(infreq_bp1_extra))
       table(b,9)= array2table(kurtosis(infreq_bp1_extra))
       table(b,10)= array2table(skewness(infreq_bp1_extra))
       table(b,11)= array2table(prctile(infreq_bp1_extra,5))
       table(b,12)= array2table(prctile(infreq_bp1_extra,95))
       table(b,13) = array2table(fd_higuchi(bp1_epoch_extra))
       env2_extra = (abs(hilbert(bp2_epoch_extra))).^2
       table(b,14)= array2table(mean(env2_extra))
       table(b,15)= array2table(std(env2_extra))
       table(b,16)= array2table(kurtosis(bp2_epoch_extra))
       table(b,17)= array2table(skewness(bp2_epoch_extra))
       table(b,18)= array2table(prctile(env2_extra,5))
       table(b,19)= array2table(prctile(env2_extra,95))
       table(b,20)= array2table(nanmean(infreq_bp2_extra))
       table(b,21)= array2table(nanstd(infreq_bp2_extra))
       table(b,22)= array2table(kurtosis(infreq_bp2_extra))
       table(b,23)= array2table(skewness(infreq_bp2_extra))
       table(b,24)= array2table(prctile(infreq_bp2_extra,5))
       table(b,25)= array2table(prctile(infreq_bp2_extra,95))
       table(b,26) = array2table(fd_higuchi(bp2_epoch_extra))
       env3_extra = (abs(hilbert(bp3_epoch_extra))).^2
       table(b,27)= array2table(mean(env3_extra))
       table(b,28)= array2table(std(env3_extra))
       table(b,29)= array2table(kurtosis(bp3_epoch_extra))
       table(b,30)= array2table(skewness(bp3_epoch_extra))
       table(b,31)= array2table(prctile(env3_extra,5))
       table(b,32)= array2table(prctile(env3_extra,95))
       table(b,33)=array2table(nanmean(infreq_bp3_extra))
       table(b,34)= array2table(nanstd(infreq_bp3_extra))
       table(b,35)= array2table(kurtosis(infreq_bp3_extra))
       table(b,36)= array2table(skewness(infreq_bp3_extra))
       table(b,37)= array2table(prctile(infreq_bp3_extra,5))
       table(b,38)= array2table(prctile(infreq_bp3_extra,95))
       table(b,39) = array2table(fd_higuchi(bp3_epoch_extra))
       env4_extra = (abs(hilbert(bp4_epoch_extra))).^2
       table(b,40)= array2table(mean(env4_extra))
       table(b,41)= array2table(std(env4_extra))
       table(b,42)= array2table(kurtosis(bp4_epoch_extra))
       table(b,43)= array2table(skewness(bp4_epoch_extra))
       table(b,44)= array2table(prctile(env4_extra,5))
       table(b,45)= array2table(prctile(env4_extra,95))
       table(b,46)= array2table(nanmean(infreq_bp4_extra))
       table(b,47)= array2table(nanstd(infreq_bp4_extra))
       table(b,48)= array2table(kurtosis(infreq_bp4_extra))
       table(b,49)= array2table(skewness(infreq_bp4_extra))
       table(b,50)= array2table(prctile(infreq_bp4_extra,5))
       table(b,51)= array2table(prctile(infreq_bp4_extra,95))
       table(b,52) = array2table(fd_higuchi(bp4_epoch_extra))
       env5_extra = (abs(hilbert(bp5_epoch_extra))).^2
       table(b,53)= array2table(mean(env5_extra))
       table(b,54)= array2table(std(env5_extra))
       table(b,55)= array2table(kurtosis(bp5_epoch_extra))
       table(b,56)= array2table(skewness(bp5_epoch_extra))
       table(b,57)= array2table(prctile(env5_extra,5))
       table(b,58)= array2table(prctile(env5_extra,95))
       table(b,59)= array2table(nanmean(infreq_bp5_extra))
       table(b,60)= array2table(nanstd(infreq_bp5_extra))
       table(b,61)= array2table(kurtosis(infreq_bp5_extra))
       table(b,62)= array2table(skewness(infreq_bp5_extra))
       table(b,63)= array2table(prctile(infreq_bp5_extra,5))
       table(b,64)= array2table(prctile(infreq_bp5_extra,95))
       table(b,65) = array2table(fd_higuchi(bp5_epoch_extra))
       table(b,66) = array2table(energy5)
       table(b,67) = array2table(energy95)
    else
       table(b,:) = []
    end 
    name = append('nirs_features', '_' ,  erase(SS(k).name, 'mydata_'))
    writetable(table, name, 'Delimiter', 'tab')
end 
