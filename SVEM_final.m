function [Lt0,Ls0,St0,Ss0,Gt0,Gs0,aLt,aLp,aSt,aSp,aGt,aGp] = SVEM_final(XDat,Ks,Kt,Kp,Dt,Dp,pct,pp)

% stochastic variational inference of spatio-temporal extreme-value graphical models
% Yu Hang, NTU, May, 2015

% Input:
% XDat (pt x ps) represents block maxima data at pt time points and ps locations
% Ks (ps x ps) describes the spatial dependence
% Kt (pt x pt) describes the temporal dependence between consecutive periods
% Kp (pt x pt) describes the temporal dependence of time points within each period
% Dt (pt x 1) is a vector of the eigenvalues of Kt
% Dp (pt x 1) is a vector of the eigenvalues of Kp
% pct (1 x 1) is the percetage of samples used to compute the stochastic graident in each iteration
% pp (1 x 1) is the period


% Output:
% Lt0 is the estimated temporal component of location parameters
% Ls0 is the estimated spatial component of location parameters
% St0 is the estimated temporal component of scale parameters
% Ss0 is the estimated spatial component of scale parameters
% Gt0 is the estimated temporal component of shape parameters
% Gs0 is the estimated spatial component of scale parameters
% aLp (1 x 1) is the log of the smoothing parameter guarding the smoothness of the
% periodic component of the location parameter
% aLt (1 x 1) is the log of the smoothing parameter guarding the smoothness of the
% trend component of the location parameter
% aSp (1 x 1) is the log of the smoothing parameter guarding the smoothness of the
% periodic component of the scale parameter
% aSt (1 x 1) is the log of the smoothing parameter guarding the smoothness of the
% trend component of the scale parameter
% aGp (1 x 1) is the smoothing parameter guarding the smoothness of the
% periodic component log of the of the shape parameter
% aGt (1 x 1) is the log of the smoothing parameter guarding the smoothness of the
% trend component of the shape parameter


%% initialization
[pt,ps] = size(XDat);
id = find(Dt+Dp>0);
Dt = Dt(id);
Dp = Dp(id);

% smoothness parameters
aLp = log(1);
aSp = log(5);
aGp = log(10);

aLt = log(5);
aSt = log(25);
aGt = log(50);

aLs = log(1);
aSs = log(5);
aGs = log(10);

% initial values of the parameters of the variational distributions
np = pt/pp;
X = reshape(XDat,pp,np,ps);
G0 = zeros(pp,ps);
S0 = zeros(pp,ps);
L0 = zeros(pp,ps);
for i = 1:pp
    for j = 1:ps
        [G0(i,j),S0(i,j),L0(i,j)]=gevfit_pwm(X(i,:,j));
    end
end
G0 = repmat(G0,np,1);
S0 = repmat(log(S0),np,1);
L0 = repmat(L0,np,1);


Gt0 = mean(G0,2); %*ones(pt,1);
Gt0 = Gt0-mean(Gt0);
Gs0 = mean(G0 - repmat(Gt0,1,ps)).';
lvGt = log(0.05*ones(pt,1));
lvGs = log(0.05*ones(ps,1));
mGt0 = Gt0;
mGs0 = Gs0;
gGt = 0;
gGs = 0;
gvGt = 0;
gvGs = 0;
g1Gt = 0;
g1Gs = 0;
g1vGt = 0;
g1vGs = 0;

St0 = mean(S0,2); %*ones(pt,1);
St0 = St0-mean(St0);
Ss0 = mean(S0 - repmat(St0,1,ps)).';
lvSt = log(0.5*ones(pt,1));
lvSs = log(0.5*ones(ps,1));
mSt0 = St0;
mSs0 = Ss0;
gSt = 0;
gSs = 0;
gvSt = 0;
gvSs = 0;
g1St = 0;
g1Ss = 0;
g1vSt = 0;
g1vSs = 0;

Lt0 = mean(L0,2); %*ones(pt,1);
Lt0 = Lt0-mean(Lt0);
Ls0 = mean(L0 - repmat(Lt0,1,ps)).';
lvLt = log(10*ones(pt,1));
lvLs = log(10*ones(ps,1));
mLt0 = Lt0;
mLs0 = Ls0;
gLt = 0;
gLs = 0;
gvLt = 0;
gvLs = 0;
g1Lt = 0;
g1Ls = 0;
g1vLt = 0;
g1vLs = 0;

Lt_array = zeros(pt,500);
Ls_array = zeros(ps,500);
St_array = zeros(pt,500);
Ss_array = zeros(ps,500);
Gt_array = zeros(pt,500);
Gs_array = zeros(ps,500);
ka = 1;

% id = [];
%% SVI algorithm

eps = 1e-6;
rho = 0.95;
eta = 0.95; %1-pct;
ns = 20; %sqrt(1/pct);
% prm_array = [];
% prmt_array = [];
% prms_array = [];
% aLp_array = [];
% ns_array = [];


nid = round(pct*ps*pt);
p = ps*pt;

ida= haltonset(1,'Skip',1e3,'Leap',1e2);
ida = scramble(ida,'RR2'); 
ida = qrandstream(ida);
idr0 = repmat((1:pt).',1,ps);
idc0 = repmat(1:ps,pt,1);

ss = 1e-3;

%% doubly stochastic variational inference
for nt = 1:3e5
    LKt = exp(aLt)*Kt+exp(aLp)*Kp;
    LKs = exp(aLs)*Ks;
    SKt = exp(aSt)*Kt+exp(aSp)*Kp;
    SKs = exp(aSs)*Ks;
    GKt = exp(aGt)*Kt+exp(aGp)*Kp;
    GKs = exp(aGs)*Ks;
    
    vLt = exp(lvLt);
    vLs = exp(lvLs);
    vSt = exp(lvSt);
    vSs = exp(lvSs);
    vGt = exp(lvGt);
    vGs = exp(lvGs);
    
    % draw samples
    
    while 1
        id = ceil(p*qrand(ida,nid));
        idr = idr0(id);
        idc = idc0(id);
    
        zLt = randn(nid,1);
        Lt_tmp = vLt(idr).*zLt+Lt0(idr);
        zLs = randn(nid,1);
        Ls_tmp = vLs(idc).*zLs+Ls0(idc);
    

        zSt = randn(nid,1);
        St_tmp = vSt(idr).*zSt+St0(idr);
        zSs = randn(nid,1);
        Ss_tmp = vSs(idc).*zSs+Ss0(idc);
    
    
        zGt = randn(nid,1);
        Gt_tmp = vGt(idr).*zGt+Gt0(idr);
        zGs = randn(nid,1);
        Gs_tmp = vGs(idc).*zGs+Gs0(idc);
    
        
        [dL0,dS0,dG0] = divlogGEV(XDat(id),Gt_tmp+Gs_tmp,St_tmp+Ss_tmp,Lt_tmp+Ls_tmp);
    
        if all(abs(dL0)<1e100) && all(abs(dS0)<1e100) && all(abs(dG0)<1e100)
            break;
        end
    end
    
    d = sparse(idr,idc,1,pt,ps);
    dL = sparse(idr,idc,dL0,pt,ps);
    dvLt = sparse(idr,idc,dL0.*zLt,pt,ps);
    dvLs = sparse(idr,idc,dL0.*zLs,pt,ps);
    dS = sparse(idr,idc,dS0,pt,ps);
    dvSt = sparse(idr,idc,dS0.*zSt,pt,ps);
    dvSs = sparse(idr,idc,dS0.*zSs,pt,ps);
    dG = sparse(idr,idc,dG0,pt,ps);
    dvGt = sparse(idr,idc,dG0.*zGt,pt,ps);
    dvGs = sparse(idr,idc,dG0.*zGs,pt,ps);
    
    
    idr = full(sum(d).');
    pr = pt./idr;
    pr(idr == 0) = 0;
    idc = full(sum(d,2));
    pc = ps./idc;
    pc(idc ==0) = 0;
    
    dLt = sum(dL,2).*pc;
    dvLt = sum(dvLt,2).*pc;
    dLs = sum(dL).'.*pr;
    dvLs = sum(dvLs).'.*pr;
    dSt = sum(dS,2).*pc;
    dvSt = sum(dvSt,2).*pc;
    dSs = sum(dS).'.*pr;
    dvSs = sum(dvSs).'.*pr;
    dGt = sum(dG,2).*pc;
    dvGt = sum(dvGt,2).*pc;
    dGs = sum(dG).'.*pr;
    dvGs = sum(dvGs).'.*pr;
    
    % compute smoothed gradients
    pr = eta*ones(pt,1);
    pr(idc == 0) = 0;
    pc = eta*ones(ps,1);
    pc(idr == 0) = 0;
    
    gLt = pr.*gLt + (1-pr).*dLt;
    gvLt = pr.*gvLt + (1-pr).*dvLt;
    gLs = pc.*gLs + (1-pc).*dLs;
    gvLs = pc.*gvLs + (1-pc).*dvLs;
    
    gSt = pr.*gSt + (1-pr).*dSt;
    gvSt = pr.*gvSt + (1-pr).*dvSt;
    gSs = pc.*gSs + (1-pc).*dSs;
    gvSs = pc.*gvSs + (1-pc).*dvSs;
    
    gGt = pr.*gGt + (1-pr).*dGt;
    gvGt = pr.*gvGt + (1-pr).*dvGt;
    gGs = pc.*gGs + (1-pc).*dGs;
    gvGs = pc.*gvGs + (1-pc).*dvGs;
    
    % compute the (stochastic) gradients for all parameters
    dvLt = (gvLt - (diag(LKt)+1).*vLt).*vLt + 1;
    dLt = gLt-LKt*Lt0-sum(Lt0);
    daLp = (sum(Dp./(exp(aLp)*Dp+exp(aLt)*Dt))/2-Lt0.'*Kp*Lt0/2-sum(diag(Kp).*vLt.^2)/2)*exp(aLp);
    daLt = (sum(Dt./(exp(aLp)*Dp+exp(aLt)*Dt))/2-Lt0.'*Kt*Lt0/2-sum(diag(Kt).*vLt.^2)/2)*exp(aLt);
    
    dvLs = (gvLs - diag(LKs).*vLs).*vLs + 1;
    dLs = gLs-LKs*Ls0;
    daLs = (ps-1)/2 - (sum(diag(Ks).*vLs.^2)+Ls0.'*Ks*Ls0)/2*exp(aLs);
    
    
    dvSt = (gvSt - (diag(SKt)+1).*vSt).*vSt + 1;
    dSt = gSt-SKt*St0-sum(St0);
    daSp = (sum(Dp./(exp(aSp)*Dp+exp(aSt)*Dt))/2-St0.'*Kp*St0/2-sum(diag(Kp).*vSt.^2)/2)*exp(aSp);
    daSt = (sum(Dt./(exp(aSp)*Dp+exp(aSt)*Dt))/2-St0.'*Kt*St0/2-sum(diag(Kt).*vSt.^2)/2)*exp(aSt);
    
    dvSs = (gvSs - diag(SKs).*vSs).*vSs + 1;
    dSs = gSs-SKs*Ss0;
    daSs = (ps-1)/2 - (sum(diag(Ks).*vSs.^2)+Ss0.'*Ks*Ss0)/2*exp(aSs);
    
    
    dvGt = (gvGt - (diag(GKt)+1).*vGt).*vGt + 1;
    dGt = gGt-GKt*Gt0-sum(Gt0);
    daGp = (sum(Dp./(exp(aGp)*Dp+exp(aGt)*Dt))/2-Gt0.'*Kp*Gt0/2-sum(diag(Kp).*vGt.^2)/2)*exp(aGp);
    daGt = (sum(Dt./(exp(aGp)*Dp+exp(aGt)*Dt))/2-Gt0.'*Kt*Gt0/2-sum(diag(Kt).*vGt.^2)/2)*exp(aGt);
    
    dvGs = (gvGs - diag(GKs).*vGs).*vGs + 1;
    dGs = gGs-GKs*Gs0;
    daGs = (ps-1)/2 - (sum(diag(Ks).*vGs.^2)+Gs0.'*Ks*Gs0)/2*exp(aGs);
    
    % determine the step size
    if nt == 1        
        g2Lt = dLt.^2; 
        g2vLt = dvLt.^2;
        g2aLp = daLp^2;
        g2aLt = daLt^2;
        g2Ls = dLs.^2;
        g2vLs = dvLs.^2;
        g2aLs = daLs^2;
        
        g2St = dSt.^2; 
        g2vSt = dvSt.^2;
        g2aSp = daSp^2;
        g2aSt = daSt^2;
        g2Ss = dSs.^2;
        g2vSs = dvSs.^2;
        g2aSs = daSs^2;
        
        g2Gt = dGt.^2; 
        g2vGt = dvGt.^2;
        g2aGp = daGp^2;
        g2aGt = daGt^2;
        g2Gs = dGs.^2;
        g2vGs = dvGs.^2;
        g2aGs = daGs^2;
    else
        g2Lt = (1-rho)*dLt.^2+rho*g2Lt; 
        g2vLt = (1-rho)*dvLt.^2+rho*g2vLt;
        g2aLp = (1-rho)*daLp^2+rho*g2aLp;
        g2aLt = (1-rho)*daLt^2+rho*g2aLt;
        g2Ls = (1-rho)*dLs.^2+rho*g2Ls;
        g2vLs = (1-rho)*dvLs.^2+rho*g2vLs;
        g2aLs = (1-rho)*daLs^2+rho*g2aLs;
        
        g2St = (1-rho)*dSt.^2+rho*g2St; 
        g2vSt = (1-rho)*dvSt.^2+rho*g2vSt;
        g2aSp = (1-rho)*daSp^2+rho*g2aSp;
        g2aSt = (1-rho)*daSt^2+rho*g2aSt;
        g2Ss = (1-rho)*dSs.^2+rho*g2Ss;
        g2vSs = (1-rho)*dvSs.^2+rho*g2vSs;
        g2aSs = (1-rho)*daSs^2+rho*g2aSs;
        
        g2Gt = (1-rho)*dGt.^2+rho*g2Gt; 
        g2vGt = (1-rho)*dvGt.^2+rho*g2vGt;
        g2aGp = (1-rho)*daGp^2+rho*g2aGp;
        g2aGt = (1-rho)*daGt^2+rho*g2aGt;
        g2Gs = (1-rho)*dGs.^2+rho*g2Gs;
        g2vGs = (1-rho)*dvGs.^2+rho*g2vGs;
        g2aGs = (1-rho)*daGs^2+rho*g2aGs;
    end
    
    g1Lt = (1-rho)*dLt+rho*g1Lt; 
    g1vLt = (1-rho)*dvLt+rho*g1vLt;
    g1Ls = (1-rho)*dLs+rho*g1Ls;
    g1vLs = (1-rho)*dvLs+rho*g1vLs;
        
    g1St = (1-rho)*dSt+rho*g1St; 
    g1vSt = (1-rho)*dvSt+rho*g1vSt;
    g1Ss = (1-rho)*dSs+rho*g1Ss;
    g1vSs = (1-rho)*dvSs+rho*g1vSs;
        
    g1Gt = (1-rho)*dGt+rho*g1Gt; 
    g1vGt = (1-rho)*dvGt+rho*g1vGt;
    g1Gs = (1-rho)*dGs+rho*g1Gs;
    g1vGs = (1-rho)*dvGs+rho*g1vGs;
    
    eta = sum(mean(g1Lt.^2)+mean(g1vLt.^2)+mean(g1Ls.^2)+mean(g1vLs.^2)+...
        mean(g1St.^2)+mean(g1vSt.^2)+mean(g1Ss.^2)+mean(g1vSs.^2)+...
        mean(g1Gt.^2)+mean(g1vGt.^2)+mean(g1Gs.^2)+mean(g1vGs.^2))/....
        sum(mean(g2Lt)+mean(g2vLt)+mean(g2Ls)+mean(g2vLs)+mean(g2St)+mean(g2vSt)+...
        mean(g2Ss)+mean(g2vSs)+mean(g2Gt)+mean(g2vGt)+mean(g2Gs)+mean(g2vGs));
    
    ns = (1-eta)*ns+1;
%     ns_array = [ns_array,ns];
    eta = 1-1/ns;


    dLt = ss./sqrt(g2Lt+eps).*dLt; 
    dvLt = ss./sqrt(g2vLt+eps).*dvLt;
    daLp = ss/sqrt(g2aLp+eps).*daLp;
    daLt = ss/sqrt(g2aLt+eps).*daLt;
    dLs = ss./sqrt(g2Ls+eps).*dLs;
    dvLs = ss./sqrt(g2vLs+eps).*dvLs;
    daLs = ss/sqrt(g2aLs+eps).*daLs;
    
    dSt = ss./sqrt(g2St+eps).*dSt; 
    dvSt = ss./sqrt(g2vSt+eps).*dvSt;
    daSp = ss/sqrt(g2aSp+eps).*daSp;
    daSt = ss/sqrt(g2aSt+eps).*daSt;
    dSs = ss./sqrt(g2Ss+eps).*dSs;
    dvSs = ss./sqrt(g2vSs+eps).*dvSs;
    daSs = ss/sqrt(g2aSs+eps).*daSs;
    
    dGt = ss./sqrt(g2Gt+eps).*dGt; 
    dvGt = ss./sqrt(g2vGt+eps).*dvGt;
    daGp = ss/sqrt(g2aGp+eps).*daGp;
    daGt = ss/sqrt(g2aGt+eps).*daGt;
    dGs = ss./sqrt(g2Gs+eps).*dGs;
    dvGs = ss./sqrt(g2vGs+eps).*dvGs;
    daGs = ss/sqrt(g2aGs+eps).*daGs;  
    
    
    
    
    % update all parameters
    Lt0 = Lt0 + dLt; 
    lvLt = lvLt + dvLt;
    aLp = aLp + daLp;
    aLt = aLt + daLt;
    Ls0 = Ls0 + dLs;
    lvLs = lvLs + dvLs;
    aLs = aLs + daLs;
    
    St0 = St0 + dSt; 
    lvSt = lvSt + dvSt;
    aSp = aSp + daSp;
    aSt = aSt + daSt;
    Ss0 = Ss0 + dSs;
    lvSs = lvSs + dvSs;
    aSs = aSs + daSs;
    
    Gt0 = Gt0 + dGt; 
    lvGt = lvGt + dvGt;
    aGp = aGp + daGp;
    aGt = aGt + daGt;
    Gs0 = Gs0 + dGs;
    lvGs = lvGs + dvGs;
    aGs = aGs + daGs;
    
    
%     prm_array = [prm_array,[Gt0(47)+Gs0(102);St0(97)+Ss0(114);Lt0(175)+Ls0(102)]];
%     prmt_array = [prmt_array,[Gt0(47);St0(97);Lt0(175)]];
%     prms_array = [prms_array,[Gs0(102);Ss0(114);Ls0(102)]];
    
    if rem(nt,10) == 0
        Lt_array(:,ka) = Lt0;
        Ls_array(:,ka) = Ls0;
        St_array(:,ka) = St0;
        Ss_array(:,ka) = Ss0;
        Gt_array(:,ka) = Gt0;
        Gs_array(:,ka) = Gs0;
        ka = ka+1;
    end
    
    if rem(nt,1000) == 0
        ss = ss*0.995;
    end
    
    % update learning rate and check convergence
    if rem(nt,5e3) == 0  %ka == 5e3; %
        mLth = mean(Lt_array,2); %Lt0; %
        mLsh = mean(Ls_array,2); %Ls0; %
        mSth = mean(St_array,2); %St0; %
        mSsh = mean(Ss_array,2); %Ss0; %
        mGth = mean(Gt_array,2); %Gt0; %
        mGsh = mean(Gs_array,2); %Gs0; %
        
        MAE_Lt = mean(abs(mLth-mLt0));
        MAE_Ls = mean(abs(mLsh-mLs0));
        MAE_St = mean(abs(mSth-mSt0));
        MAE_Ss = mean(abs(mSsh-mSs0));
        MAE_Gt = mean(abs(mGth-mGt0));
        MAE_Gs = mean(abs(mGsh-mGs0));
        fprintf('#iterations =%d,MAE_Lt=%d,MAE_Ls=%d,MAE_St=%d,MAE_Ss=%d,MAE_Gt=%d,MAE_Gs=%d\n',nt,MAE_Lt,MAE_Ls,MAE_St,MAE_Ss,MAE_Gt,MAE_Gs);       
        
        %save pr;
        
        Gest = repmat(Gt0,1,ps)+repmat(Gs0.',pt,1);
        Sest = exp(repmat(St0,1,ps)+repmat(Ss0.',pt,1));
        Lest = repmat(Lt0,1,ps)+repmat(Ls0.',pt,1);
        a2 = 1+Gest./Sest.*(XDat-Lest);
        a3 = sum(a2(:)<0);
        
        if MAE_Lt<1e-2  && MAE_Ls<1e-2 && ...    
                MAE_St<1e-2 && MAE_Ss<1e-2  && ...
                MAE_Gt<1e-2  && MAE_Gs<1e-2 && a3 == 0 
                break;
        else
            mLt0 = mLth;
            mLs0 = mLsh;
            mSt0 = mSth;
            mSs0 = mSsh;
            mGt0 = mGth;
            mGs0 = mGsh;   
            ka = 1;
        end
    end
end


% save(['pr',num2str(pct),'_final.mat']);