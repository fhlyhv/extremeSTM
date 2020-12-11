function [dL,dS,dG] = divlogGEV(XDat,G,S,L)

% compute the dirivatives of GEV distributions for the three models

G(G==0) = 1e-6;

a1 = (XDat-L)./exp(S);
a2 = 1+a1.*G;
a3 = (1+G)./a2;
a4 = a2.^(-1./G-1);
a5 = a3-a4;

dL = a5./exp(S);
dL(a2<=0) = -1e10*G(a2<=0)./exp(S(a2<=0));

dS = -1+a5.*a1;
dS(a2<=0) = -1e10*G(a2<=0).*a1(a2<=0);

dG = (1-a2.^(-1./G)).*log(a2)./G.^2-a5.*a1./G;
dG(a2<=0) = 1e10*a1(a2<=0);

% dL(isinf(dL)|isinf(dS)|isinf(dG)|G==0) = 0;
% dS(isinf(dL)|isinf(dS)|isinf(dG)|G==0) = 0;
% dG(isinf(dL)|isinf(dS)|isinf(dG)|G==0) = 0;