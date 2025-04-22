%% Code for FBCPM
clc;
clear;
%% Narrowband Functianal Connectivity
fs = 256;

freqR = [1,45];
nfreq = freqR(2)-freqR(1);
epoch_length = 0.5; % 0-500 ms from stimulation onset 
nchan = 60;
nSubjcet = 137;

pvt_duration = 15; % select 0-10 min PVT data
pvt_matrix = [];

for sidx = 1:nSubject 
    pvt_epoch = [];
    for fidx = 1:nfreq
        EEG_filt{fidx} = pop_eegfiltnew(EEG,'locutoff',freqR(1)+fidx-1,'hicutoff',freqR(1)+fidx,'plotfreqz',0);
    end
    for fidx = 1:nfreq
        EEG_data = EEG_filt{fidx}.data;
        % Given the epoch onset for each subject 
        for i = 1:length(epoch)
            pvt_epoch(:,:,fidx,i) = EEG_data(:,epoch(i):epoch(i)+epoch_length*fs-1);
        end
    end
    
    % Calculate PLI
    Mpvt = zeros(nchan,nchan,nfreq,size(pvt_epoch,4));
    for fidx = 1:nfreq
        disp(['Freq',num2str(fidx),': ---------- Subject ',num2str(sidx),' ----------'])
        for i = 1:size(pvt_epoch,4)
            Mpvt(:,:,fidx,i) = extractFC(pvt_epoch(:,:,fidx,i),'PLI');
        end
    end
    pvt_matrix(:,:,:,sidx) = mean(Mpvt,4);
end

%% CPM framework 
% Given the mean RT during the PVT task -- behaviour_result
% and the narrowband FC -- pvt_matrix

[~,nchan,nfreq,nSubject] = size(pvt_matrix);

fs_all = [0.05,0.01,0.005];
nFS = length(fs_all);

behav_pred_pos = zeros(nSubject,nFS);
behav_pred_neg = zeros(nSubject,nFS);
behav_pred_both = zeros(nSubject,nFS);

for leftout = 1:nSubject
    fprintf('Leaving out subject # %1.0f \n',leftout);
    train_mats = pvt_matrix;
    test_mats = train_mats(:,:,:,leftout);
    train_mats(:,:,:,leftout) = [];

    train_behav = behaviour_result;
    test_behav = train_behav(leftout,:);
    train_behav(leftout,:) = [];
    
    for fs_idx = 1:nFS
        fs_thre = fs_all(fs_idx);
        % correlate all edges with behaviour result     
        [r_mat,p_mat] = corr(reshape(train_mats,[],nSubject-1)',train_behav);
        r_mat = reshape(r_mat,nchan,nchan,nfreq);
        p_mat = reshape(p_mat,nchan,nchan,nfreq);

        pos_mask = zeros(nchan,nchan,nfreq);
        neg_mask = zeros(nchan,nchan,nfreq);

        pos_edges = find(r_mat>0 & p_mat<fs_thre);
        neg_edges = find(r_mat<0 & p_mat<fs_thre);
        pos_mask(pos_edges) = 1;
        neg_mask(neg_edges) = 1;
        
        train_sumpos = zeros(nSubject-1,1);
        train_sumneg = zeros(nSubject-1,1);

        con = ones(nSubject-1,1);

        for ss = 1:nSubject-1
            train_sumpos(ss,1) = sum(train_mats(:,:,:,ss).*pos_mask,'all')/2;
            train_sumneg(ss,1) = sum(train_mats(:,:,:,ss).*neg_mask,'all')/2;
        end
        test_sumpos = sum(test_mats.*pos_mask,'all')/2;
        test_sumneg = sum(test_mats.*neg_mask,'all')/2;

        train_sumboth = cat(2,train_sumpos,train_sumneg);
        test_sumboth = cat(2,test_sumpos,test_sumneg);
        
        fit_pos = regress(train_behav,[con,train_sumpos]);
        behav_pred_pos(leftout,fs_idx) = fit_pos(1)+fit_pos(2:end)'*test_sumpos;

        fit_neg = regress(train_behav,[con,train_sumneg]);
        behav_pred_neg(leftout,fs_idx) = fit_neg(1)+fit_neg(2:end)'*test_sumneg;

        fit_both = regress(train_behav,[con,train_sumboth]);
        behav_pred_both(leftout,fs_idx) = fit_both(1)+fit_both(2:end)'*test_sumboth;
    end
end

R_all = []; MAE_all = [];
for fs_idx = 1:nFS
    R_corr = [] ;
    [R_corr(1,1),~] = corr(behaviour_result,reshape(behav_pred_pos(:,fs_idx),[],1));
    [R_corr(1,2),~] = corr(behaviour_result,reshape(behav_pred_neg(:,fs_idx),[],1));
    [R_corr(1,3),~] = corr(behaviour_result,reshape(behav_pred_both(:,fs_idx),[],1)); 
    R_all = [R_all,R_corr];
    MAE = [];
    MAE(1,1) = mean(abs(behaviour_result-reshape(behav_pred_pos(:,fs_idx),[],1)));
    MAE(1,2) = mean(abs(behaviour_result-reshape(behav_pred_neg(:,fs_idx),[],1)));
    MAE(1,3) = mean(abs(behaviour_result-reshape(behav_pred_both(:,fs_idx),[],1)));
    MAE_all = [MAE_all,MAE];
end
