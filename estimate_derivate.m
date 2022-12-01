function y = estimate_derivate(x)
%---------------------------------------------------------------------
% Estimate the derivate using either forward finite difference
% approximation or an FIR-based approximation.
%---------------------------------------------------------------------
USE_DIFF = 0;


if(USE_DIFF)
    % 1. first-order estimate:
    y = diff(x);

else
    % 2. FIR filter:    
    L_h = floor(length(x) / 4);

    
    % some contraints on the length of the filter:
    if(L_h > 50)
        L_h = 50;
    end
    % want a Type-IV filter (as length = L_h + 1)
    if(rem(L_h, 2))
        L_h = L_h - 1;
    end

    % use least-squares approach to approximate ideal filter-response:
    b = firls(L_h, [0 0.9], [0 0.9 * pi], 'differentiator');
    y = filter(b, 1, x);    

    % fvtool(b, 1, 'MagnitudeDisplay', 'zero-phase');
end