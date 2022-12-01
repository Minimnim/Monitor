P = 'C:\Users\PhysioUser\Desktop\PhD\Monitor\Minoo\pre-processed signals\Transients';
S = dir(fullfile(P,'*.txt')); 
PP = 'C:\Users\PhysioUser\Desktop\PhD\Monitor\Minoo\pre-processed signals\NIRS containing transients\mydata';
SS = dir(fullfile(PP,'*.txt')); 
for k = 1:numel(S)
    clear table
    F = fullfile(P,S(k).name);
    S(k).data = readtable(F);
    transient = table2array(S(k).data)
    FF = fullfile(PP,SS(k).name);
    SS(k).data = readtable(FF);
    mydata = table2array(SS(k).data)
    sat = mydata(:,2)
    transient(transient >= mode(transient)) = NaN
    total_pow = nanmean(abs(transient).^2 )
    transient = table2array(S(k).data)
    transient(transient> mode(transient)) = mode(transient)
    env = (abs(hilbert(transient))).^2
    meanenv = mean(env)
    SDenv = std(env)
    m0 = nanvar(transient)
    activity = m0
    x_hat = estimate_derivate(transient);
    m2 = nanvar(x_hat);
    mobility = sqrt(m2 / m0);
    x_hat2 = estimate_derivate(x_hat);
    m4 = nanvar(x_hat2);
    mobility2 = sqrt(m4 / m2);
    complexity = mobility2 / mobility;
    %to measure inter spike interval
    transient(transient< mode(transient)) = "NaN"
    y = length(transient)
%     transient(7738,:) = NaN
%     transient(16919,:) = mode(transient)
    b = zeros(y,1)
    if isnan(transient(1,1)) == 1
        for x = 2:y-1
            a = find((isnan(transient(x,1)) == 1) && (isnan(transient(x-1,1)) == 0))
            if a == 1
                b(x,1) = a
            end   
        end
        for x = 2:y-1
            c = find((isnan(transient(x,1)) == 1) && (isnan(transient(x+1,1)) == 0))
            if c == 1
                b(x,1) = c
            end 
        end
    d = find(b==1)
    time = mydata(:,1)
    time = [time(1,1); time(d); time(end,1)]
    time = diff(time)
    isi = time(2:2:end,:)
    else
        for x = 1:y-1
            a = find((isnan(transient(x,1)) == 1) && (isnan(transient(x-1,1)) == 0))
            if a == 1
                b(x,1) = a
            end   
        end
        for x = 1:y-1
            c = find((isnan(transient(x,1)) == 1) && (isnan(transient(x+1,1)) == 0))
            if c == 1
                b(x,1) = c
            end 
        end
        d = find(b==1)
        time = mydata(:,1)
        time = [time(1,1); time(d); time(end,1)]
        time = diff(time)
        isi = time(1:2:end,:)
    end 
    %to check if the right thing has been captured 
    b = zeros(y,1)
    for x = 2:y-1
        g = find((isnan(transient(x-1,1)) == 1) && (isnan(transient(x+1,1)) == 1) && (isnan(transient(x,1)) == 0))
        if g == 1
            b(x,1) = g
        end 
    end 
    for x = 2:y-1
        g = find((isnan(transient(x-1,1)) == 0) && (isnan(transient(x+1,1)) == 0) && (isnan(transient(x,1)) == 1))
        if g == 1
            b(x,1) = g
        end 
    end 
    if find(b == 1) ~= 0 
        error('error in ISI')
    end
    %to measure the time spent below 63%
    transient = table2array(S(k).data)
    transient(transient< 63) = "NaN"
%     transient(17029,:) = 63.0570
    y = length(transient)
    b = zeros(y,1)
    if isnan(transient(1,1)) == 1
        for x = 2:y-1
            a = find((isnan(transient(x,1)) == 1) && (isnan(transient(x-1,1)) == 0))
            if a == 1
            b(x,1) = a
            end   
        end
        for x = 2:y-1
            c = find((isnan(transient(x,1)) == 1) && (isnan(transient(x+1,1)) == 0))
            if c == 1
            b(x,1) = c
            end 
        end
        d = find(b==1)
        time = mydata(:,1)
        time = [time(1,1); time(d); time(end,1)]
        time = diff(time)
        below63 = time(3:2:end,:)
        below63 = sum(below63)
    else 
        for x = 1:y-1
            a = find((isnan(transient(x,1)) == 1) && (isnan(transient(x-1,1)) == 0))
            if a == 1
                b(x,1) = a
            end   
        end
        for x = 1:y-1
            c = find((isnan(transient(x,1)) == 1) && (isnan(transient(x+1,1)) == 0))
            if c == 1
                b(x,1) = c
            end 
        end
        d = find(b==1)
        time = mydata(:,1)
        time = [time(1,1); time(d); time(end,1)]
        time = diff(time)
        below63 = time(2:2:end,:)
        below63 = sum(below63)
    end 
    %to check if the right thing has been captured 
    y = length(transient)
    b = zeros(y,1)
    for x = 2:y-1
        g = find((isnan(transient(x-1,1)) == 1) && (isnan(transient(x+1,1)) == 1) && (isnan(transient(x,1)) == 0))
        if g == 1
            b(x,1) = g
        end 
    end 
    for x = 2:y-1
        g = find((isnan(transient(x-1,1)) == 0) && (isnan(transient(x+1,1)) == 0) && (isnan(transient(x,1)) == 1))
        if g == 1
            b(x,1) = g
        end 
    end 
    if find(b == 1) ~= 0 
        error('error in 63%')
    end
    %to measure the time spent below 63% for raw rcSO2
    sat = fillmissing(sat, 'previous')
    sat(sat<= 63) = "NaN"
    y = length(sat)
    b = zeros(y,1)
    if isnan(sat(1,1)) == 1 
        for x = 2:y-1
            a = find((isnan(sat(x,1)) == 1) && (isnan(sat(x-1,1)) == 0))
            if a == 1
                b(x,1) = a
            end   
        end
        for x = 2:y-1
            c = find((isnan(sat(x,1)) == 1) && (isnan(sat(x+1,1)) == 0))
            if c == 1
                b(x,1) = c
            end 
        end
        d = find(b==1)
        time = mydata(:,1)
        time = [time(1,1); time(d); time(end,1)]
        time = diff(time)
        below63_rcSO2 = time(3:2:end,:)
        below63_rcSO2 = sum(below63_rcSO2)
    else 
        for x = 1:y-1
            a = find((isnan(sat(x,1)) == 1) && (isnan(sat(x-1,1)) == 0))
            if a == 1
                b(x,1) = a
            end   
        end 
        for x = 1:y-1
            c = find((isnan(sat(x,1)) == 1) && (isnan(sat(x+1,1)) == 0))
            if c == 1
                b(x,1) = c
            end 
        end
        d = find(b==1)
        time = mydata(:,1)
        time = [time(1,1); time(d); time(end,1)]
        time = diff(time)
        below63_rcSO2 = time(2:2:end,:)
        below63_rcSO2 = sum(below63_rcSO2)
    end 
    %to check if the right thing has been captured 
    y = length(sat)
    b = zeros(y,1)
    for x = 2:y-1
        g = find((isnan(sat(x-1,1)) == 1) && (isnan(sat(x+1,1)) == 1) && (isnan(sat(x,1)) == 0))
        if g == 1
            b(x,1) = g
        end 
    end 
    for x = 2:y-1
        g = find((isnan(sat(x-1,1)) == 0) && (isnan(sat(x+1,1)) == 0) && (isnan(sat(x,1)) == 1))
        if g == 1
            b(x,1) = g
        end 
    end
    if find(b == 1) ~= 0 
        error('error in 63% in main signal')
    end
    %to calculate bandwidth ans slope
    transient = table2array(S(k).data)
    time = mydata(:,1)
    base = mode(transient)
    transient(transient>= base) = "NaN"
%     transient(7738,:) = mode(transient)
%     transient(16919,:) = NaN
    y = length(transient)
    b = zeros(y,1)
    if isnan(transient(1,1)) == 1
        for x = 1:y-1
            a = find((isnan(transient(x,1)) == 0) && (isnan(transient(x-1,1)) == 1))
            if a == 1
                b(x,1) = a
            end   
        end 
        for x = 1:y-1
            c = find((isnan(transient(x,1)) == 0) && (isnan(transient(x+1,1)) == 1))
            if c == 1
                b(x,1) = c
            end 
        end
        d = find(b==1)
        starts = d(1:2:end)
        stops = d(2:2:end)
%         starts(end,:)=[]
        if length(starts) ~= length(stops)
            error('starts and stops are not equal')
        end
        longRuns = stops - starts
        long = max(longRuns)+1
        a = zeros(long,1)
        newcol = zeros(long,1)
        for x = 1:length(starts)
            a(:,x) = [transient(starts(x):stops(x)); zeros(((long-1) - longRuns(x)),1)]
            a = [a newcol]
        end
        numcol = size(a,2)
        a = a(:, 1:(numcol-1))
        a(a==0) = NaN
        [y x] = nanmin(a)
        amp = base - y
        slope_down = (y - base)./x
        slope_up = (base-y)./(length(a)-sum(isnan(a))-x)
        time = mydata(:,1)
        time = [time(1,1); time(d); time(end,1)]
        time = diff(time)
        width = time(2:2:end,:)
    else
        for x = 2:y-1
            a = find((isnan(transient(x,1)) == 0) && (isnan(transient(x-1,1)) == 1))
            if a == 1
            b(x,1) = a
            end   
        end 
        for x = 2:y-1
            c = find((isnan(transient(x,1)) == 0) && (isnan(transient(x+1,1)) == 1))
            if c == 1
                b(x,1) = c
            end 
        end
        d = find(b==1)
        starts = d(2:2:end)
        stops = d(3:2:end)
        if length(starts) ~= length(stops)
            error('starts and stops are not equal')
            print(F)
        end 
        %starts(end,:)=[]
        longRuns = stops - starts
        long = max(longRuns)+1
        a = zeros(long,1)
        newcol = zeros(long,1)
        for x = 1:length(starts)
            a(:,x) = [transient(starts(x):stops(x)); zeros(((long-1) - longRuns(x)),1)]
            a = [a newcol]
        end
        numcol = size(a,2)
        a = a(:, 1:(numcol-1))
        a(a==0) = NaN
        [y x] = nanmin(a)
        amp = base - y
        slope_down = (y - base)./x
        slope_up = (base-y)./(length(a)-sum(isnan(a))-x)
        time = mydata(:,1)
        time = [time(1,1); time(d); time(end,1)]
        time = diff(time)
        width = time(3:2:end,:)
    end 
    %to check if the right thing has been captured 
    y = length(transient)
    b = zeros(y,1)
    for x = 2:y-1
        g = find((isnan(transient(x-1,1)) == 1) && (isnan(transient(x+1,1)) == 1) && (isnan(transient(x,1)) == 0))
        if g == 1
            b(x,1) = g
        end 
    end 
    for x = 2:y-1
        g = find((isnan(transient(x-1,1)) == 0) && (isnan(transient(x+1,1)) == 0) && (isnan(transient(x,1)) == 1))
        if g == 1
            b(x,1) = g
        end 
    end 
    g = find((isnan(transient(x-1,1)) == 0) && (isnan(transient(x+1,1)) == 0) && (isnan(transient(x,1)) == 1))
    if find(b == 1) ~= 0 
        error('error in bandwidth')
        print(F)
    end
    %to combine all the features in one table and save it as text file
    table = table(total_pow, meanenv, SDenv, activity, mobility, complexity, mean(isi), std(isi), below63, mean(amp), mean(slope_down.'), mean(slope_up.'), mean(width), length(starts), below63_rcSO2)
    name = append('transient_features', '_' ,  erase(S(k).name, 'mydata_'))
    writetable(table, name, 'Delimiter', 'tab')
end 