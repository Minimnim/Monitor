function [y_ssa]=iterative_SSA_decomposition(x,L,method,all_Ls,use_dct,dct_keep,db_plot)
if(nargin<2 || isempty(L)), L=20; end
if(nargin<3 || isempty(method)), method='Vautard_Ghil'; end
if(nargin<4 || isempty(all_Ls)), all_Ls=[25:5:40]; end
if(nargin<4 || isempty(use_dct)), use_dct=1; end
if(nargin<5 || isempty(dct_keep)), dct_keep=0.1; end
if(nargin<7 || isempty(db_plot)), db_plot=0; end

DBverbose=0;

% important to keep mean of original signal:
xmean=nanmean(x);

%---------------------------------------------------------------------
% first pass:
%---------------------------------------------------------------------
subtract_mean=true;
[y_decomp,S]=noise_reduction_SSA(x,L,method,use_dct,dct_keep,subtract_mean,[],db_plot); 

% if no iterative component:
if(isempty(all_Ls))
    y_ssa=y_decomp;
    return;
end


%---------------------------------------------------------------------
% iterative over all L
%---------------------------------------------------------------------
y_iter=y_decomp; 
frac_LS=S/L;
subtract_mean=false;
for n=1:length(all_Ls)
    
    % set limit of max. S
    L_max=round(all_Ls(n)*frac_LS);

    if(DBverbose)
        fprintf('L=%d (L-max=%d)\n',all_Ls(n),L_max);
    end
    
    
    ymean=nanmean(y_iter);

    [y_iter,S]=noise_reduction_SSA(y_iter-ymean,all_Ls(n),method, ...
                                   use_dct,1,subtract_mean,L_max,false); 
    
    
    frac_LS=S/all_Ls(n);
    
    % important, sum back original mean:
    y_iter=y_iter+xmean;
end


y_ssa=y_iter;


db_plot = false;
if(db_plot)
    set_figure(44);
    subplot(2,1,1); hold all;
    plot(x); plot(y_ssa); 
    legend({'x + transient', 'iter. SSA decomp.'});
    subplot(2,1,2); hold all;
    plot(x); plot(x - y_ssa + xmean);
    legend({'x', 'x - iter. SSA decomp.'});
end
