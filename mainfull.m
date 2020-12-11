% fixed-form variational inference for spatio-temporal model
% Yu Hang, Jun, 2014, NTU

clear;

%% read data

load artiData(16x16)_GSL; 

PMdat = XDat(1:end-12,:); % 
[pt,ps] =size(PMdat);  % pt -- no. of time instants  ps -- no. of sites 

grid = 'regular';  % specify the grid type
% grid = 'irregular';

%%  specify graphical model structure
pp = 12; % period of the time series, e.g., pp = 12 for monthly data

if strcmp(grid,'regular')
    % regular grid
    pc = 16; % no. of columns in the grid
    pr = 16; % no. of rows in the grid % pr x pc grid
    [Ks,Kt,Kp,Kp0,Ds,Dt,Dp] = constructKsKtKp(pr,pc,pt,pp);
else
    % irregular grid
    Ltt = 10*rand(200,1);  % a column vector of latitudes
    Lgt = 10*rand(200,1);  % a column vector of longitudes
    [Ks,Kt,Kp,Kp0,Ds,Dt,Dp] = constructKsKtKp_irregular(Ltt,Lgt,pt,pp);
end

%% variational EM inference
tic;
[Lt0,Ls0,St0,Ss0,Gt0,Gs0,aLt,aLp,aSt,aSp,aGt,aGp] = SVEM_final(PMdat,Ks,Kt,Kp,Dt,Dp,0.05,pp);
t = toc

% estimation of the GEV parameters as pt x ps matrices
Gest = repmat(Gt0,1,ps)+repmat(Gs0.',pt,1);
Sest = exp(repmat(St0,1,ps)+repmat(Ss0.',pt,1));
Lest = repmat(Lt0,1,ps)+repmat(Ls0.',pt,1);


%% prediction of the GEV paremeter in the next period
Gtp = (exp(aGp)*Kp0+exp(aGt)*speye(pp))\(2*Gt0(end-pp+1:end)-Gt0(end-2*pp+1:end-pp))*exp(aGt);
Stp = (exp(aSp)*Kp0+exp(aSt)*speye(pp))\(2*St0(end-pp+1:end)-St0(end-2*pp+1:end-pp))*exp(aSt);
Ltp = (exp(aLp)*Kp0+exp(aLt)*speye(pp))\(2*Lt0(end-pp+1:end)-Lt0(end-2*pp+1:end-pp))*exp(aLt);

Gpredict = repmat(Gtp,1,ps)+repmat(Gs0.',pp,1);
Spredict = exp(repmat(Stp,1,ps)+repmat(Ss0.',pp,1));
Lpredict = repmat(Ltp,1,ps)+repmat(Ls0.',pp,1);