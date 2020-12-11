function [G,S,L]=gevfit_pwm(X0)

n=length(X0);
X=sort(X0);

beta0=mean(X0);
beta1=0;
beta2=0;

for i=1:n
    beta1=beta1+1/n*(i-1)/(n-1)*X(i);
    beta2=beta2+1/n*(i-1)*(i-2)/(n-1)/(n-2)*X(i);
end

% syms k;
% G=double(solve((3^k-1)/(2^k-1)-(3*beta2-beta0)/(2*beta1-beta0)));
c=(2*beta1-beta0)/(3*beta2-beta0)-log(2)/log(3);
G=-(7.8590*c+2.9554*c^2);
S=(2*beta1-beta0)*G/gamma(1-G)/(2^G-1);
L=beta0+S/G*(1-gamma(1-G));