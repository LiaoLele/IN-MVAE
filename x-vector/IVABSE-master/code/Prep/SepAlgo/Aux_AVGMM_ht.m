function [ time] = Aux_AVGMM_ht(x, cfgs, varargin)
% required varargin:
% - pre_step: 0（没有对观测信号进行预白化）或 1（对观测信号预白化）
% - state_num


if ischar(cfgs)
    run(cfgs);
else
    varnames = fieldnames(cfgs);
    for ii = 1 : length(varnames)
        eval([varnames{ii}, '= getfield(cfgs, varnames{ii});']);
    end 
end

if exist('varargin', 'var')
    for ii = 1 : 2 : length(varargin)
        eval([varargin{ii}, '= varargin{ii+1};'])
    end
end

pObj = Inf;
tol = 1e-6;
nol = 3*frame_len/4;

%% STFT
len_x = length(x);
s_sep = zeros(2,len_x);
Y_ori = gmm_STFT(x.', frame_len, win.', frame_move);
X = zeros(size(Y_ori,1), size(Y_ori,2), sou_num);
Y_ori(frame_len_half+1:end,:,:) = [];
if pre_step == 1
    Y_ori_shp = permute(Y_ori,[3,2,1]);
    [Y,Q] = pre_whitening_ori(Y_ori);  
else
    Y = Y_ori;
end
frame_num = size(Y, 2);     
Y_shp = permute(Y,[3,2,1]);  

%% variables initialization
W = zeros(sou_num, ch_num, frame_len_half);
w_tmp = W;
WY = zeros(sou_num,frame_num,frame_len_half);
h = zeros(sou_num,frame_num);
% h = zeros(state_num,frame_num,sou_num);
for j = 1:frame_len_half
    W(:,:,j) = eye(sou_num);
    WY(:,:,j) = W(:,:,j)*Y_shp(:,:,j);
end
pophi = zeros(state_num,frame_len_half,sou_num);
for i = 1:sou_num
     if hpophi == 0
         h(i,:) = unifrnd(0.999,1.001,1,frame_num);
     else
         pophi(:,:,i) = unifrnd(0.999,1.001,state_num,frame_len_half); 
     end
end
p_s = 1/state_num^2*ones(state_num); 
Q_st = zeros(state_num,state_num,frame_num);
for i = 1:frame_num
   Q_st(:,:,i) = 1/state_num^2*ones(state_num); 
end
WY_squ2 = abs(WY).^2;
WY_squ = permute(WY_squ2,[3,2,1]);
part = zeros(state_num,frame_num,2);
Q1 = sum(Q_st,2);
Q1 = reshape(Q1,state_num,frame_num,1);
Q2 = sum(Q_st,1);
Q2 = reshape(Q2,state_num,frame_num,1);
pophi_ij_save = zeros(frame_len_half,frame_num,state_num,2);

%% EM algorithm

for time = 1:maxiter
    % 更新时域激励 h_ori_ori
        h(1,:) = frame_len_half./(sum((pophi(:,:,1).'*Q1).*WY_squ(:,:,1),1)+eps);
        h(2,:) = frame_len_half./(sum((pophi(:,:,2).'*Q2).*WY_squ(:,:,2),1)+eps);   
    
    for m = 1:2
        for i = 1:state_num
            pophi_ij = pophi(i,:,m).'*h(m,:) + eps;  % F*T
%             pophi_ij = pophi(i,:,m).'*h(i,:,m) + eps;  % F*T
            pophi_ij_save(:,:,i,m) = pophi_ij;
            part1 = sum(log(pophi_ij),1);
            part2 = sum(pophi_ij.*WY_squ(:,:,m),1);
            part(i,:,m) = part1 - part2;             
        end    
    end
    sou1 = repmat(reshape(part(:,:,1),state_num,1,frame_num),1,state_num,1);
    sou2 = repmat(reshape(part(:,:,2),1,state_num,frame_num),state_num,1,1);
    logGaussProb = sou1 + sou2;

    mark = find(p_s == 0);
    if (isempty(mark) == 0)
       for ii = 1:frame_num
           tmp = logGaussProb(:,:,ii);
           tmp(mark) = min(min(tmp));
           logGaussProb(:,:,ii) = tmp;
       end
    end

    logGaussProb2 = logGaussProb - max(max(logGaussProb,[],1),[],2);
    logGaussProb3 = exp(logGaussProb2).*repmat(p_s,1,1,frame_num);
    Q_st = logGaussProb3./sum(sum(logGaussProb3,1),2);  
    log_pg = max(log(p_s),-1e-100)+logGaussProb;
    
    Q1 = sum(Q_st,2);
    Q1 = reshape(Q1,state_num,frame_num,1);
    Q2 = sum(Q_st,1);
    Q2 = reshape(Q2,state_num,frame_num,1);
    QQ(:,:,1) = Q1;
    QQ(:,:,2) = Q2;
    
    % 更新滤波器系数 Decouple的方法
    for n = 1:2
        preci = permute(pophi_ij_save(:,:,:,n),[3,2,1]);
        preci = permute(sum(QQ(:,:,n).*preci,1),[3,2,1]);
        Y_mod = permute(preci.*Y,[3,2,1]);
        for k = 1:frame_len_half
            V = Y_mod(:,:,k)*Y_shp(:,:,k)'/frame_num;
            V(1,1) = real(V(1,1));
            V(2,2) = real(V(2,2));
            WV_inv = inv(W(:,:,k)*V);
            w_tmp(:,n,k) = WV_inv(:,n);
            w_tmp(:,n,k) = w_tmp(:,n,k)*(w_tmp(:,n,k)'*V*w_tmp(:,n,k))^(-0.5);        
            WY(n,:,k) = w_tmp(:,n,k)'*Y_shp(:,:,k);
        end   
        mean_x = sqrt(mean(mean(abs(WY(n,:,:)).^2,2),3));
        w_tmp(:,n,:) = w_tmp(:,n,:)./mean_x;
        WY(n,:,:) = WY(n,:,:)./mean_x;
    end  
    W = conj(permute(w_tmp,[2,1,3]));
    WY_squ = abs(WY).^2;
    WY_squ = permute(WY_squ,[3,2,1]);
    
    % 更新频域模态    
        pophi(:,:,1) = bsxfun(@rdivide, sum(Q1,2), Q1*(h(1,:).*WY_squ(:,:,1)).'+eps);
        pophi(:,:,2) = bsxfun(@rdivide, sum(Q2,2), Q2*(h(2,:).*WY_squ(:,:,2)).'+eps);

    % err: 考察辅助函数Q(theta)的变化
    Obj = sum(sum(sum(log_pg.*Q_st,1),2),3);
    dObj(time) = abs(Obj-pObj)/abs(Obj);
    pObj = Obj;
    if dObj(time) < tol, break;end
    
    if mod(time,10) == 0
        fprintf('%d iterations: Objective=%e, dObj=%e\n',time,Obj,dObj(time));
    end
    % 更新先验概率
    p_s = mean(Q_st,3);   
end   


%% 生成分离后信号
if pre_step == 1
    for m = 1:2
         E = zeros(2,2);
         E(m,m) = 1;
         for k = 1:frame_len_half
             obs = (W(:,:,k)*Q(:,:,k)^(-0.5))\(E*W(:,:,k)*Q(:,:,k)^(-0.5))*Y_ori_shp(:,:,k);
             X(k,:,m) = obs(1,:);
         end 
    end
else
    for m = 1:2
         E = zeros(2,2);
         E(m,m) = 1;
         for k = 1:frame_len_half
             obs = (W(:,:,k))\(E*W(:,:,k))*Y_shp(:,:,k);
             X(k,:,m) = obs(1,:);
         end 
    end
end
X(frame_len_half+1:end,:,:) = conj(X(frame_len_half-1:-1:2,:,:));
for i = 1:sou_num
    s_sep(i,:) = gmm_ISTFT(X(:,:,i), len_x, win.', frame_move);
    s_sep(i,:) = s_sep(i,:)/max(abs(s_sep(i,:))); 
end
