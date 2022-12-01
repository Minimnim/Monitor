P = 'C:\Users\PhysioUser\Desktop\PhD\Monitor\Minoo\pre-processed signals\Transients';
S = dir(fullfile(P,'*.txt')); 
PP = 'C:\Users\PhysioUser\Desktop\PhD\Monitor\Minoo\pre-processed signals\NIRS containing transients\mydata';
SS = dir(fullfile(PP,'*.txt')); 
i = zeros(65,1)
for k = 1:numel(S)
    clear table
    F = fullfile(P,S(k).name);
    S(k).data = readtable(F);
    transient = table2array(S(k).data)
    FF = fullfile(PP,SS(k).name);
    SS(k).data = readtable(FF);
    mydata = table2array(SS(k).data)
    sat = mydata(:,2)
    %to measure the time spent below 63% for raw rcSO2
    sat = fillmissing(sat, 'previous')
    sat(sat<= 63) = "NaN"
    %to fix the NaN value between two numbers, or one number between two
    %NaN values
    y = length(sat)
    b = zeros(y,1)
    for x = 2:y-1
        g = find((isnan(sat(x-1,1)) == 1) && (isnan(sat(x+1,1)) == 1) && (isnan(sat(x,1)) == 0))
        if g == 1
            sat(x,1) = NaN
        end 
    end 
    for x = 2:y-1
        g = find((isnan(sat(x-1,1)) == 0) && (isnan(sat(x+1,1)) == 0) && (isnan(sat(x,1)) == 1))
        if g == 1
            sat(x,1) = sat(x-1, 1) 
        end 
    end
    if find(b == 1) ~= 0 
        disp(k)
        error('error in 63% in main signal')
    end
    b = zeros(y,1)
    if isnan(sat(1,1)) == 1 && isnan(sat(2,1)) == 0
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
        below63_rcSO2 = time(2:2:end,:)
        below63_rcSO2 = sum(below63_rcSO2)
    end 
    if isnan(sat(1,1)) == 1 && isnan(sat(2,1)) == 1
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
    end 
    if isnan(sat(1,1)) == 0
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
    i(k,1) = below63_rcSO2
end 