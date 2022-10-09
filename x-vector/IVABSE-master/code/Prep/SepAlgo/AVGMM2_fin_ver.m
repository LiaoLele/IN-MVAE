% -------------AV-GMM-IVA(FIN_VER)--------------
% activation parameter h is related to frame index only
% 2019-02-24

clc;
% close all;
clear all;
rng(0)

%% global variables
fs = 16e3;
frame_len = 64e-3*fs;
frame_len_half = frame_len/2+1;
frame_move = frame_len/4;
win = hanning(frame_len).';
sou_num = 2;
ch_num = 2;
maxiter = 100;
state_num = 2;
pObj = Inf;
tol = 1e-6;

%% audioreading
% **********************voa*****************************
sou_path_name = 'primarynoise.wav';%'C:\Users\Administrator\Desktop\20190807ANC实验\primarynoise.wav';
mix_path_name = 'RefCal.wav';%'C:\Users\Administrator\Desktop\20190807ANC实验\RefCal.wav';
[x,f] = audioread(mix_path_name);x = resample(x,fs,f);x=10*x.';
[s,f] = audioread(sou_path_name);s = resample(s,fs,f);s = s.';

MixNum = 2;
% %{
%% STFT
len_x = length(x);
s_sep = zeros(MixNum,len_x);
Y_ori = gmm_STFT(x.', frame_len, win.', frame_move);
X = zeros(size(Y_ori,1), size(Y_ori,2), sou_num);
Y_ori(frame_len_half+1:end,:,:) = [];
[Y,Q] = pre_whitening_ori(Y_ori);   
frame_num = size(Y, 2);     
Y_con = permute(Y,[3,1,2]);
Y_con = reshape(Y_con, MixNum*frame_len_half, frame_num, 1);

%% variables initialization
W = zeros(sou_num, ch_num, frame_len_half);
h = zeros(sou_num, frame_num);
M = zeros(2,2,frame_len_half);
for j = 1:frame_len_half
    W(:,:,j) = eye(sou_num);
end
pophi = zeros(state_num,frame_len_half,sou_num);
a = ceil(frame_num/2);
for i = 1:sou_num
     pophi(:,:,i) = unifrnd(0.999,1.001,state_num,frame_len_half);  
% h(i,:) = unifrnd(0.999,1.001,1,frame_num);
% pophi(:,:,i) = ones(state_num,frame_len_half);  
%     pophi(1,:,i) = 1./mean((abs(Y(:,1:a,i))).^2,2);
%     pophi(2,:,i) = 1./mean((abs(Y(:,a+1:end,i))).^2,2);
%  h(i,:) = ones(1,frame_num);
end
p_s = 1/state_num^2*ones(state_num); 
Q_st = zeros(state_num,state_num,frame_num);
for i = 1:frame_num
   Q_st(:,:,i) = 1/state_num^2*ones(state_num); 
end
W_cell = mat2cell(W,MixNum,MixNum,ones(1,frame_len_half));
W_blk = blkdiag(W_cell{:});
WY = W_blk*Y_con;
WY_squ = abs(WY).^2;

%% EM algorithm
for time = 1:maxiter
%     time
    logGaussProb = zeros(state_num,state_num,frame_num);
    % 更新时域激励 h_ori_ori
    h(1,:) = frame_len_half./(sum((pophi(:,:,1).'*reshape(sum(Q_st,2),state_num,frame_num,1)).*WY_squ(1:2:end,:),1)+eps);
    h(2,:) = frame_len_half./(sum((pophi(:,:,2).'*reshape(sum(Q_st,1),state_num,frame_num,1)).*WY_squ(2:2:end,:),1)+eps);
%     Q1 = sum(Q_st,2);
%     Q1 = reshape(Q1,state_num,frame_num,1);
%     Q2 = sum(Q_st,1);
%     Q2 = reshape(Q2,state_num,frame_num,1);
%     pophi(:,:,1) = bsxfun(@rdivide, sum(Q1,2), Q1*(h(1,:).*WY_squ(1:2:end,:)).'+eps);
%     pophi(:,:,2) = bsxfun(@rdivide, sum(Q2,2), Q2*(h(2,:).*WY_squ(2:2:end,:)).'+eps);

% 更新后验概率
    for i = 1:state_num
        for j = 1:state_num
            for k = 1:frame_len_half
                pophi_ijk = bsxfun(@times, [pophi(i,k,1);pophi(j,k,2)], [h(1,:);h(2,:)])+eps;
                logGaussProb(i,j,:) = logGaussProb(i,j,:) + ...
                    reshape(-sum(real(conj(WY(2*(k-1)+1:2*k,:)).*WY(2*(k-1)+1:2*k,:)).*pophi_ijk)+sum(log(pophi_ijk)),1,1,frame_num);
            end
        end
    end
%     logGaussProb2 = logGaussProb + repmat(log(p_s),1,1,frame_num);
%     logGaussProb2 = reshape(logGaussProb2,state_num^2,1,frame_num);
%     logGaussProb3 = logGaussProb2 - permute(logGaussProb2,[2,1,3]);
%     logGaussProb4 = 1./sum(exp(logGaussProb3),1);
%     Q_st = reshape(logGaussProb4,state_num,state_num,frame_num);
    
    
    logGaussProb2 = logGaussProb - max(max(logGaussProb,[],1),[],2);
    logGaussProb3 = exp(logGaussProb2).*repmat(p_s,1,1,frame_num);
    Q_st = logGaussProb3./sum(sum(logGaussProb3,1),2);  
    log_pg = log(p_s)+logGaussProb;

%     % 更新频域模态
%     Q1 = sum(Q_st,2);
%     Q1 = reshape(Q1,state_num,frame_num,1);
%     Q2 = sum(Q_st,1);
%     Q2 = reshape(Q2,state_num,frame_num,1);
%     pophi(:,:,1) = bsxfun(@rdivide, sum(Q1,2), Q1*(h(1,:).*WY_squ(1:2:end,:)).'+eps);
%     pophi(:,:,2) = bsxfun(@rdivide, sum(Q2,2), Q2*(h(2,:).*WY_squ(2:2:end,:)).'+eps);
    
    % 更新滤波器系数
    for k = 1:frame_len_half
        vk1 = reshape(pophi(:,k,1).*h(1,:), state_num,1,frame_num);
        vk2 = reshape(pophi(:,k,2).*h(2,:), 1,state_num, frame_num);
        vk1 = repmat(vk1, 1,state_num,1);
        vk2 = repmat(vk2, state_num,1,1);
        gain = sum(sum(Q_st.*(vk1-vk2),1),2);
        Y_M = reshape(gain,1,frame_num,1).*Y_con(2*(k-1)+1:2*k,:);        
        M(:,:,k) = Y_M*Y_con(2*(k-1)+1:2*k,:)';
        M_ass = M(:,:,k);
        M_ass(1,1) = real(M_ass(1,1));
        M_ass(2,2) = real(M_ass(2,2));
        beta = (M_ass(1,1)+M_ass(2,2))/2 - sqrt(((M_ass(1,1)-M_ass(2,2))/2)^2 + abs(M_ass(1,2))^2);
        a_star(k) = 1/sqrt(1 + abs((beta-M_ass(1,1))/M_ass(1,2))^2);
        b_star(k) = ((beta-M_ass(1,1))/M_ass(1,2))*a_star(k);  
        W(1,1,k) = conj(a_star(k));
        W(2,2,k) = a_star(k);
        W(1,2,k) = conj(b_star(k));
        W(2,1,k) = -b_star(k);
    end  
    W_cell = mat2cell(W,2,2,ones(1,frame_len_half));
    W_blk = blkdiag(W_cell{:});
    WY = W_blk*Y_con;
    WY_squ = abs(WY).^2;
    
    % 更新频域模态
    Q1 = sum(Q_st,2);
    Q1 = reshape(Q1,state_num,frame_num,1);
    Q2 = sum(Q_st,1);
    Q2 = reshape(Q2,state_num,frame_num,1);
    pophi(:,:,1) = bsxfun(@rdivide, sum(Q1,2), Q1*(h(1,:).*WY_squ(1:2:end,:)).'+eps);
    pophi(:,:,2) = bsxfun(@rdivide, sum(Q2,2), Q2*(h(2,:).*WY_squ(2:2:end,:)).'+eps);
%     h(1,:) = frame_len_half./(sum((pophi(:,:,1).'*reshape(sum(Q_st,2),state_num,frame_num,1)).*WY_squ(1:2:end,:),1)+eps);
%     h(2,:) = frame_len_half./(sum((pophi(:,:,2).'*reshape(sum(Q_st,1),state_num,frame_num,1)).*WY_squ(2:2:end,:),1)+eps);

    % err: 考察辅助函数Q(theta)的变化
    Obj = sum(sum(sum(log_pg.*Q_st,1),2),3);
    dObj(time) = abs(Obj-pObj)/abs(Obj);
    pObj = Obj;
    if dObj(time) < tol, break;end
    
    if mod(time,10) == 0
        fprintf('%d iterations: Objective=%e, dObj=%e\n',time,Obj,dObj(time));
    end
    
%     for p = 1:frame_len_half
%         X_tmp = W(:,:,p)*Y_con(2*(p-1)+1:2*p,:);
%         % spectral compensation 
%         X(p,:,:) = permute(diag(diag(Q(:,:,p)^(0.5)/(W(:,:,p))))*X_tmp,[3,2,1]);
%         W_GMM(:,:,p) = W(:,:,p)*Q(:,:,p)^(-0.5);
%         W_GMM(:,:,p) = diag(diag(pinv(W_GMM(:,:,p))))*W_GMM(:,:,p);
%     end
%     X(frame_len_half+1:end,:,:) = conj(X(frame_len_half-1:-1:2,:,:));
%     for i = 1:sou_num
%         s_sep(i,:) = gmm_ISTFT(X(:,:,i), len_x, win.', frame_move);
%         s_sep(i,:) = s_sep(i,:)/max(abs(s_sep(i,:))); 
%     end
%     [sdr,sir,sar,~] = bss_eval_sources(s_sep, s)
    
    % 更新先验概率
    p_s = mean(Q_st,3);
end      

%% 生成分离后信号
for p = 1:frame_len_half
    X_tmp = W(:,:,p)*Y_con(2*(p-1)+1:2*p,:);
    % spectral compensation 
    X(p,:,:) = permute(diag(diag(Q(:,:,p)^(0.5)/(W(:,:,p))))*X_tmp,[3,2,1]);
    W_GMM(:,:,p) = W(:,:,p)*Q(:,:,p)^(-0.5);
    W_GMM(:,:,p) = diag(diag(pinv(W_GMM(:,:,p))))*W_GMM(:,:,p);
end
X(frame_len_half+1:end,:,:) = conj(X(frame_len_half-1:-1:2,:,:));
for i = 1:sou_num
    s_sep(i,:) = gmm_ISTFT(X(:,:,i), len_x, win.', frame_move);
    s_sep(i,:) = s_sep(i,:);%/max(abs(s_sep(i,:))); 
end
[sdr,sir,sar,~] = bss_eval_sources(s_sep, s)
% 初始SIR
[sdr_ori,sir_ori,sar_ori,~] = bss_eval_sources(x, s)
% end
% audiowrite('ref2.wav',s_sep.',fs);
% audiowrite('test1_offline_10s_2_2.wav',s_sep(1,:).',fs);
% audiowrite('test2_offline_10s_2_2.wav',s_sep(2,:).',fs);
%}
sir_imp = sir-sir_ori

%% postprocessing

[Ref_sep,f] = audioread('RefCal_sep.wav');Ref_sep = resample(Ref_sep,fs,f);Ref_sep=10*Ref_sep.';

RefNum = 2;
figure;
for i = 1:RefNum
    subplot(2,3,(i-1)*3+1);hold on;
    [pxx,f] = pwelch(Ref_sep(i,:),boxcar(fs),fs/2,fs,fs);
    plot(2*f/fs,10*log10(pxx'),'r');
    [pxx,f] = pwelch(s_sep(i,:),boxcar(fs),fs/2,fs,fs);
    plot(2*f/fs,10*log10(pxx'),'b');hold off;%normalized freq
    legend('Perfectly Seperated Signal','BSS Seperated Signal');
    xlabel('Normalized Freq');ylabel('Power Spectrum Density / dB');
%     plot(f,10*log10(pxx'),'Color',[i/2,i/3,1]);
    subplot(2,3,(i-1)*3+2);
    hold on;plot(s_sep(i,:));plot(Ref_sep(i,:),'r');hold off
    legend('BSS Seperated Signal','Perfectly Seperated Signal');
    subplot(2,3,i*3);
    mscohere(s_sep(i,:),Ref_sep(i,:));
end
audiowrite('sep.wav', s_sep.'/10, fs);

%% IVA
%{
% frame_len = nfft;
[y, W_iva] = fivabss_true(x, frame_len);
for i = 1:sou_num
    y(i,:) = y(i,:)/max(abs(y(i,:))); 
end
[sdr,sir,sar,~] = bss_eval_sources(y, s)
% audiowrite('test.wav', y.', fs)
% [sdr,sir,sar,~] = bss_eval_sources(YY, s)
% % audiowrite('test1_mixed_0.15_120_50_2_f_3_f.wav',y(1,:).',fs);
% % audiowrite('test2_mixed_0.15_120_50_2_f_3_f.wav',y(2,:).',fs);
% audiowrite('test1_mixed_0.15_120_50_2_f_3_f_0.1.wav',YY(1,:).',fs);
% audiowrite('test2_mixed_0.15_120_50_2_f_3_f_0.1.wav',YY(2,:).',fs);
%}
%% plot 
% 指向性图
% frame_len = 128e-3*fs;
% frame_move = frame_len/4;
% addpath(genpath('directivity_fig'));
% cfgs = 'directivity_fig/configs_dicfig.m';
% run(cfgs);
% fig_directivity(W5,frame_len,dicplt,'fs',fs,'r_dis',0.1,'frame_len',frame_len,'frame_move',frame_move)
% % fig_directivity(W_iva,frame_len,dicplt,'fs',fs,'r_dis',0.1,'frame_len',frame_len,'frame_move',frame_move)
% rmpath(genpath('directivity_fig'));
% 
% figure;
% for i = 1:state_num
%     subplot(1,state_num,i)
%     plot(pophi(i,:,1),(1:frame_len_half)*fs/frame_len);
% end
% figure;
% for i = 1:state_num
%     subplot(1,state_num,i)
%     plot(pophi(i,:,2),(1:frame_len_half)*fs/frame_len);
% end
% 
% figure;
% plot((1:frame_num)*0.032,h(1,:));
% figure;
% plot((1:frame_num)*0.032,h(2,:));
% 
% 
% 

