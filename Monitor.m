%to open and read the NIRS files in matlab; 
table0 = readtable('171125N_MN0001.R10','FileType', 'text')
table1 = readtable('171126N_MN0001.R11','FileType', 'text')
data = [table0; table1]
%to change time to seconds starting from zero
a = deftime(data);
b = defdate(data); 
ts = time2sec(a,b);
%to plot the data before artefact removal; Sat in stored in column called RSO2
figure;hold on;grid on
sat = data.Var16
plot(ts, sat, 'b.');
%artefact removal; Since the device can only record the values between
%15-95%, the values less than 15% are artefact and should be removed with a
%collar of 30 seconds around it (15 seconds before and 15 seconds after 15%) 
tfifteens = ts(find(sat==15));
collar = defcollar(tfifteens);
remove = ismember(ts, collar);
data = table(ts, sat);
data(remove,:)=[];
%to rewrite the updated sat and ts 
[ts, sat] = renew(data);
%to plot the data before next step begins 
plot(ts,sat, 'y.');
%to remove NaNs at the start and end of signal; the number of rows differs
%in each participant, and some participants do not have any NaN value 
%this file has to be divided into 2 parts because of lots of NaN values in
%between. the first file consists of: 
%data = data(217:4820,:)
%the second part consists of:
% data = data(5304:end,:)
%To rewrite and plot the updated ts and sat before next step begins
[ts, sat] = renew(data);
plot(ts, sat, 'g.');
%to find where rcSO2=95% 
tninetyfive = ts(find(sat==95));
%to replcase NaNs with cubic spline interpolation
tnans = ts(find(sat == 0))
sat(sat==0)=NaN; 
data = table(ts, sat);
nansreplaced = fillmissing(data, 'pchip');
%To rewrite and plot the updated ts and sat before next step begins
[ts, sat] = renew(nansreplaced);
plot(ts, sat, 'r.');
%to upsample to 10 Hz 
Fs = 10; 
last_time = ceil(ts(end)/Fs)*Fs
time_to_interp = ts(1) : 1/Fs : last_time 
upsampled_data = [time_to_interp ; interp1(ts, sat, time_to_interp , 'pchip')]
%To rewrite and plot the updated ts and sat before next step begins
ts = upsampled_data(1,:)
sat = upsampled_data(2,:)
plot(ts, sat, 'b.')
%to anti-alising filter of 1/12 Hz. John recommended this method for
%antialising for preventing numerical instability because of low order
[z p k] = butter(6, (1/12)/(10/2),'low')
[sos, g] = zp2sos(z,p,k)
antiali = filtfilt(sos,g,sat) 
plot(ts, antiali, 'r.');
%to downsample to 1/6 Hz
time_to_interp = ts(1) : 6 : last_time
downsampled_data = [time_to_interp ; interp1(ts, antiali, time_to_interp)]
%To rewrite and plot the updated ts and sat before next step begins
ts = downsampled_data(1,:)
sat = downsampled_data(2,:)
plot(ts, sat, 'g.')
sat = sat.'
ts = ts.'
%to fill in NaNs that were replaced from step 3 
[minvalue, closestindex]= min(abs(ts-tnans.'));
sat(closestindex,:)=NaN;
%to fill in 95% values identified in step 2 
[minvalue1, closestindex1]= min(abs(ts-tninetyfive.'));
sat(closestindex1,:)=95;
plot(ts, sat, 'b')
%to save the pre-processed data as mydata.txt; if the NIRS file is broken to seperate files with a rather long duration between files, and have been analyzed 
%in different parts, there would be more than one mydata.txt; EX:
%mydata1.txt, mydata2.txt & ...
data = table(ts, sat)
writetable(data, 'mydata.txt', 'Delimiter', 'tab')