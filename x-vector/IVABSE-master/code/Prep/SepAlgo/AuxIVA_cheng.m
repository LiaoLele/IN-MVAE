function [y, W] = AuxIVA_cheng(x, nfft, maxiter, tol, ns)

[nx,nn]=size(x);
if ~exist('nfft','var')||isempty(nfft), nfft=1024; end
if ~exist('maxiter','var')||isempty(maxiter), maxiter=100; end
if ~exist('tol','var')||isempty(tol), tol=1e-6; end
if ~exist('ns','var')||isempty(ns), ns=nx; end

win=2*hanning(nfft,'periodic');
shift=fix(3*nfft/4);
for i=1:nx
    X(i,:,:)=conj(stft(x(i,:)',nfft,win,shift)');
end
N=size(X,2);
nf=size(X,3);

pObj=Inf;
W=zeros(ns,ns,nf);
Wp=zeros(ns,ns,nf);
Q=zeros(ns,nx,nf);
Xp=zeros(ns,N,nf);
S=zeros(ns,N,nf);
Si=zeros(1,N,nf);
Vi=zeros(ns,ns,nf);
y=zeros(ns,nn);

for k=1:nf
    Xmean=mean(X(:,:,k),2)*ones(1,N);
    Rxx=(X(:,:,k)-Xmean)*(X(:,:,k)-Xmean)'/N;
    [E,D]=eig(Rxx);
    d=real(diag(D));
    [tmp,order]=sort(-d);
    E=E(:,order(1:ns));
    D=diag(real(d(order(1:ns)).^(-1/2)));
    Q(:,:,k)=D*E';
    Xp(:,:,k)=Q(:,:,k)*(X(:,:,k)-Xmean);
    Wp(:,:,k)=eye(ns);
end

for iter=1:maxiter
    dlw=0;
    for i=1:ns
        Wi=Wp(:,i,:);
        for k=1:nf
            Si(:,:,k)=Wi(:,:,k)'*Xp(:,:,k);
        end
        ri=sum(abs(Si).^2,3).^0.5;
        ei=zeros(ns,1); ei(i)=1;
        for k=1:nf
            Vi(:,:,k)=(Xp(:,:,k)./(ones(ns,1)*ri+eps))*Xp(:,:,k)'/N;
            Wi(:,:,k)=(Wp(:,:,k)'*Vi(:,:,k))\ei;
            Wi(:,:,k)=Wi(:,:,k)./(sqrt(Wi(:,:,k)'*Vi(:,:,k)*Wi(:,:,k))+eps);
        end
        Wp(:,i,:)=Wi;    
    end
    for k=1:nf
        dlw=dlw+log(abs(det(Wp(:,:,k)))+eps);
    end
%     Obj=(-dlw)/(ns*nf);
%     dObj=pObj-Obj;
%     pObj=Obj;
%     if mod(iter,10)==0
%         fprintf('%d iterations: Obj=%e, dObj=%e\n',iter,Obj,dObj);
%     end
%     if abs(dObj)/abs(Obj)<tol
%         break;
%     end
end
% disp(iter);
for k=1:nf
    W(:,:,k)=Wp(:,:,k)'*Q(:,:,k);
    W(:,:,k)=diag(diag(pinv(W(:,:,k))))*W(:,:,k);
    S(:,:,k)=W(:,:,k)*X(:,:,k);
end

for i=1:ns
    y(i,:)=istft(conj(squeeze(S(i,:,:))'),nn,win,shift)';
    y(i,:) = y(i,:)/max(abs(y(i,:)));
end

end

