function [Lt,Ls,St,Ss,Gt,Gs,aLt,aLp,aLs,aSt,aSp,aSs,aGt,aGp,aGs] = Gibbssampling(XDat,Ks,Kt,Kp,Dt,Dp,pp,nall)

% Learning spatio-temporal model using Gibbs sampling
% Yu Hang, NTU, Mar, 2015

%% initialization

% parametes of hyper gamma priors (non-informative)
a = 1e-6;
b = 1e-6;
c = 1e-6;
d = 1e-6;
e = 1e-6;
f = 1e-6;

[pt,ps] = size(XDat);
np = floor(pt/pp);
id = find(Dt+Dp>0);
Dt = Dt(id);
Dp = Dp(id);

aLp = zeros(nall,1);
aLp(1) = 1;
aSp = zeros(nall,1);
aSp(1) = 1;
aGp = zeros(nall,1);
aGp(1) = 2;
maLp0 = aLp(1);
maSp0 = aSp(1);
maGp0 = aGp(1);

aLt = zeros(nall,1);
aLt(1) = 2;
aSt = zeros(nall,1);
aSt(1) = 5;
aGt = zeros(nall,1);
aGt(1) = 5;
maLt0 = aLt(1);
maSt0 = aSt(1);
maGt0 = aGt(1);

aLs = zeros(nall,1);
aLs(1) = 1;
aSs = zeros(nall,1);
aSs(1) = 2;
aGs = zeros(nall,1);
aGs(1) = 5;
maLs0 = aLs(1);
maSs0 = aSs(1);
maGs0 = aGs(1);

Gt = zeros(nall,pt);
Gs = zeros(nall,ps);
St = zeros(nall,pt);
Ss = zeros(nall,ps);
Lt = zeros(nall,pt);
Ls = zeros(nall,ps);

Prm = gevfit(XDat(:));
G0 = Prm(1);
S0 = Prm(2);
L0 = Prm(3);
L0 = 0.2*L0 + 0.8*XDat;
S0 = log(S0);
Gt(1,:) = 1e-4*ones(1,pt);
Gs(1,:) = G0*ones(1,ps);
mGt0 = Gt(1,:);
mGs0 = Gs(1,:);

% St(1,:) = S0/2*ones(1,pt);
Ss(1,:) = S0*ones(1,ps);
mSt0 = St(1,:);
mSs0 = Ss(1,:);

Lt(1,:) = mean(L0,2).'/2; %*ones(pt,1);
Ls(1,:) = mean(L0)/2+mean(Lt(1,:)); %*ones(ps,1);
Lt(1,:) = Lt(1,:)-mean(Lt(1,:));
mLt0 = Lt(1,:);
mLs0 = Ls(1,:);

ntt = 5e3;
%% Gibbs sampling

tic;
for nt = 2:nall
 
    GKs = aGs(nt-1)*Ks;
    SKs = aSs(nt-1)*Ks;
    LKs = aLs(nt-1)*Ks;
    
    Gs(nt,:) = Gs(nt-1,:);
    Ss(nt,:) = Ss(nt-1,:);
    Ls(nt,:) = Ls(nt-1,:);
    Gt(nt,:) = Gt(nt-1,:);
    St(nt,:) = St(nt-1,:);
    Lt(nt,:) = Lt(nt-1,:);
    
    for i = 1:ps
        cid = setdiff(1:ps,i);
        Gstmp = normrnd(Gs(nt,i),min(sqrt(2/GKs(i,i)),0.03));
        rq = gevpdf(XDat(:,i).',Gt(nt,:)+Gstmp,exp(St(nt,:)+Ss(nt,i)),Lt(nt,:)+Ls(nt,i))./...
            gevpdf(XDat(:,i).',Gt(nt,:)+Gs(nt,i),exp(St(nt,:)+Ss(nt,i)),Lt(nt,:)+Ls(nt,i));
%         rq(isnan(rq)) = 1;
        q = min(1,prod(rq)*exp((Gs(nt,i)-Gstmp)*sum(GKs(i,cid).*Gs(nt,cid))+(Gs(nt,i)^2-Gstmp^2)*GKs(i,i)/2));
        if rand(1) < q
            Gs(nt,i) = Gstmp;
        end
        
        
        Sstmp = normrnd(Ss(nt,i),min(sqrt(2/SKs(i,i)),0.3));
        rq = gevpdf(XDat(:,i).',Gt(nt,:)+Gs(nt,i),exp(St(nt,:)+Sstmp),Lt(nt,:)+Ls(nt,i))./...
            gevpdf(XDat(:,i).',Gt(nt,:)+Gs(nt,i),exp(St(nt,:)+Ss(nt,i)),Lt(nt,:)+Ls(nt,i));
%         rq(isnan(rq)) = 1;
        q= min(1,prod(rq)*exp((Ss(nt,i)-Sstmp)*sum(SKs(i,cid).*Ss(nt,cid))+(Ss(nt,i)^2-Sstmp^2)*SKs(i,i)/2));
        if rand(1) < q
            Ss(nt,i) = Sstmp;
        end
        
        
        Lstmp = normrnd(Ls(nt,i),min(sqrt(2/LKs(i,i)),3));
        rq = gevpdf(XDat(:,i).',Gt(nt,:)+Gs(nt,i),exp(St(nt,:)+Ss(nt,i)),Lt(nt,:)+Lstmp)./...
            gevpdf(XDat(:,i).',Gt(nt,:)+Gs(nt,i),exp(St(nt,:)+Ss(nt,i)),Lt(nt,:)+Ls(nt,i));
%         rq(isnan(rq)) = 1;
        q = min(1,prod(rq)*exp((Ls(nt,i)-Lstmp)*sum(LKs(i,cid).*Ls(nt,cid))+(Ls(nt,i)^2-Lstmp^2)*LKs(i,i)/2));   %normpdf(q,mLs,vLs)/normpdf(Ls(nt,i),mLs,vLs));
        if rand(1) < q
            Ls(nt,i) = Lstmp;
        end
    end
        
    aGs(nt) = gamrnd(a+(ps-1)/2,1/(b+Gs(nt,:)*Ks*Gs(nt,:).'/2));
    aSs(nt) = gamrnd(a+(ps-1)/2,1/(b+Ss(nt,:)*Ks*Ss(nt,:).'/2));
    aLs(nt) = gamrnd(a+(ps-1)/2,1/(b+Ls(nt,:)*Ks*Ls(nt,:).'/2));
        
    GKt = aGt(nt-1)*Kt+aGp(nt-1)*Kp+ones(pt);
    SKt = aSt(nt-1)*Kt+aSp(nt-1)*Kp+ones(pt);
    LKt = aLt(nt-1)*Kt+aLp(nt-1)*Kp+ones(pt);
    

    
    for i = 1:pt
        cid = setdiff(1:pt,i);
        Gttmp = normrnd(Gt(nt,i),min(sqrt(2/GKt(i,i)),0.03));
        rq = gevpdf(XDat(i,:),Gs(nt,:)+Gttmp,exp(Ss(nt,:)+St(nt,i)),Ls(nt,:)+Lt(nt,i))./...
            gevpdf(XDat(i,:),Gs(nt,:)+Gt(nt,i),exp(Ss(nt,:)+St(nt,i)),Ls(nt,:)+Lt(nt,i));
%         rq(isnan(rq)) = 1;
        q = min(1,prod(rq)*exp((Gt(nt,i)-Gttmp)*sum(GKt(i,cid).*Gt(nt,cid))+(Gt(nt,i)^2-Gttmp^2)*GKt(i,i)/2));
        if rand(1) < q
            Gt(nt,i) = Gttmp;
        end
        
        
        Sttmp = normrnd(St(nt,i),min(sqrt(2/SKt(i,i)),0.3));
        rq = gevpdf(XDat(i,:),Gs(nt,:)+Gt(nt,i),exp(Ss(nt,:)+Sttmp),Ls(nt,:)+Lt(nt,i))./...
            gevpdf(XDat(i,:),Gs(nt,:)+Gt(nt,i),exp(Ss(nt,:)+St(nt,i)),Ls(nt,:)+Lt(nt,i));
%         rq(isnan(rq)) = 1;
        q = min(1,prod(rq)*...
            exp((St(nt,i)-Sttmp)*sum(SKt(i,cid).*St(nt,cid))+(St(nt,i)^2-Sttmp^2)*SKt(i,i)/2));
        if rand(1) < q
            St(nt,i) = Sttmp;
        end
        
        Lttmp = normrnd(Lt(nt,i),min(sqrt(2/LKt(i,i)),3));
        rq = gevpdf(XDat(i,:),Gs(nt,:)+Gt(nt,i),exp(Ss(nt,:)+St(nt,i)),Ls(nt,:)+Lttmp)./...
            gevpdf(XDat(i,:),Gs(nt,:)+Gt(nt,i),exp(Ss(nt,:)+St(nt,i)),Ls(nt,:)+Lt(nt,i));
%         rq(isnan(rq)) = 1;
        q = min(1,prod(rq)*exp((Lt(nt,i)-Lttmp)*sum(LKt(i,cid).*Lt(nt,cid))+(Lt(nt,i)^2-Lttmp^2)*LKt(i,i)/2));
        if rand(1) < q
            Lt(nt,i) = Lttmp;
        end
    end
    
    bb = 1/(d+Gt(nt,:)*Kp*Gt(nt,:).'/2);
    aGptmp = gamrnd(aGp(nt-1)/bb,bb);
    q = min(1,exp((c-1)*(log(aGptmp)-log(aGp(nt-1)))-(d+Gt(nt,:)*Kp*Gt(nt,:).'/2)*(aGptmp-aGp(nt-1))+...
        sum(log(aGptmp*Dp+aGt(nt-1)*Dt))/2-sum(log(aGp(nt-1)*Dp+aGt(nt-1)*Dt))/2)*(gampdf(aGptmp,aGp(nt-1)/bb,bb)/gampdf(aGp(nt-1),aGptmp/bb,bb)));
    if rand(1) < q
        aGp(nt) = aGptmp;
    else
        aGp(nt) = aGp(nt-1);
    end
    
    bb = 1/(f+Gt(nt,:)*Kt*Gt(nt,:).'/2);
    aGttmp = gamrnd(aGt(nt-1)/bb,bb);
    q = min(1,exp((e-1)*(log(aGttmp)-log(aGt(nt-1)))-(f+Gt(nt,:)*Kt*Gt(nt,:).'/2)*(aGttmp-aGt(nt-1))+...
        sum(log(aGp(nt)*Dp+aGttmp*Dt))/2-sum(log(aGp(nt)*Dp+aGt(nt-1)*Dt))/2)*(gampdf(aGttmp,aGt(nt-1)/bb,bb)/gampdf(aGt(nt-1),aGttmp/bb,bb)));
    if rand(1) < q
        aGt(nt) = aGttmp;
    else
        aGt(nt) = aGt(nt-1);
    end
    
    bb = 1/(d+St(nt,:)*Kp*St(nt,:).'/2);
    aSptmp = gamrnd(aSp(nt-1)/bb,bb);
    q = min(1,exp((c-1)*(log(aSptmp)-log(aSp(nt-1)))-(d+St(nt,:)*Kp*St(nt,:).'/2)*(aSptmp-aSp(nt-1))+...
        sum(log(aSptmp*Dp+aSt(nt-1)*Dt))/2-sum(log(aSp(nt-1)*Dp+aSt(nt-1)*Dt))/2)*(gampdf(aSptmp,aSp(nt-1)/bb,bb)/gampdf(aSp(nt-1),aSptmp/bb,bb)));
    if rand(1) < q
        aSp(nt) = aSptmp;
    else
        aSp(nt) = aSp(nt-1);
    end
    
    bb = 1/(f+St(nt,:)*Kt*St(nt,:).'/2);
    aSttmp = gamrnd(aSt(nt-1)/bb,bb);
    q = min(1,exp((e-1)*(log(aSttmp)-log(aSt(nt-1)))-(f+St(nt,:)*Kt*St(nt,:).'/2)*(aSttmp-aSt(nt-1))+...
        sum(log(aSp(nt)*Dp+aSttmp*Dt))/2-sum(log(aSp(nt)*Dp+aSt(nt-1)*Dt))/2)*(gampdf(aSttmp,aSt(nt-1)/bb,bb)/gampdf(aSt(nt-1),aSttmp/bb,bb)));
    if rand(1) < q
        aSt(nt) = aSttmp;
    else
        aSt(nt) = aSt(nt-1);
    end
    
    bb = 1/(d+Lt(nt,:)*Kp*Lt(nt,:).'/2);
    aLptmp = gamrnd(aLp(nt-1)/bb,bb);
    q = min(1,exp((c-1)*(log(aLptmp)-log(aLp(nt-1)))-(d+Lt(nt,:)*Kp*Lt(nt,:).'/2)*(aLptmp-aLp(nt-1))+...
        sum(log(aLptmp*Dp+aLt(nt-1)*Dt))/2-sum(log(aLp(nt-1)*Dp+aLt(nt-1)*Dt))/2)*(gampdf(aLptmp,aLp(nt-1)/bb,bb)/gampdf(aLp(nt-1),aLptmp/bb,bb)));
    if rand(1) < q
        aLp(nt) = aLptmp;
    else
        aLp(nt) = aLp(nt-1);
    end
    
    bb = 1/(f+Lt(nt,:)*Kt*Lt(nt,:).'/2);
    aLttmp = gamrnd(aLt(nt-1)/bb,bb);
    q = min(1,exp((e-1)*(log(aLttmp)-log(aLt(nt-1)))-(f+Lt(nt,:)*Kt*Lt(nt,:).'/2)*(aLttmp-aLt(nt-1))+...
        sum(log(aLp(nt)*Dp+aLttmp*Dt))/2-sum(log(aLp(nt)*Dp+aLt(nt-1)*Dt))/2)*(gampdf(aLttmp,aLt(nt-1)/bb,bb)/gampdf(aLt(nt-1),aLttmp/bb,bb)));
    if rand(1) < q
        aLt(nt) = aLttmp;
    else
        aLt(nt) = aLt(nt-1);
    end    
    
end  

