function [Ks,Kt,Kp,Kp0,Ds,Dt,Dp] = constructKsKtKp(pr,pc,pt,pp)

% construct structure matrix for the three components of GEV parameters
% Yu Hang, NTU, Jul. 2014


% spatial pattern
if pr > 1 || pc > 1
    Kr = spdiags([ones(pc,1),zeros(pc,1),ones(pc,1)],-1:1,pc,pc);
    Kr = -Kr+spdiags(sum(Kr,2),0,pc,pc);
    Kc = spdiags([ones(pr,1),zeros(pr,1),ones(pr,1)],-1:1,pr,pr);
    Kc = -Kc+spdiags(sum(Kc,2),0,pr,pr);
    Ks = kron(speye(pr),Kr)+kron(Kc,speye(pc));
    Ks = Ks*Ks;
    [~,Ds] = eigs(Ks,pr*pc-1);
    Ds = [diag(Ds);0];
else
    Ks = 1;
    Ds = 1;
end


% trend pattern
if pt > 1
    Kp0 = spdiags([-1*ones(pp,1),-1*ones(pp,1),2*ones(pp,1),-1*ones(pp,1),-1*ones(pp,1)],[-pp+1,-1:1,pp-1],pp,pp);
    Kp0 = Kp0'*Kp0;
    [~,Dp] = eigs(Kp0,pp-1);
    Dp = [diag(Dp);0];
    np = pt/pp;
    Kp = kron(speye(np),Kp0);
    Dp = kron(ones(np,1),Dp);

    Kt = spdiags([-ones(np,1),2*ones(np,1),-ones(np,1)],0:2,np-2,np);
    Kt = Kt'*Kt;
    [~,Dt] = eigs(Kt,np-2);
    Dt = [diag(Dt);zeros(2,1)];
    Kt = kron(Kt,speye(pp));
    Dt = kron(Dt,ones(pp,1));
else
    Kp = 1;
    Kp0 = 1;
    Dp = 1;
    Kt = 1;
    Dt = 1;
end

    