function FC_matrix = extractFC(EEG,method)

% input: 
% EEG: channel*time EEG data
% method: optional method ==> "PLI"

% output:
% FC_matrix: adjacency matrix corresponding to selected method

nchan = size(EEG,1);
FC_matrix = zeros(nchan,nchan);

for i = 1:nchan
    for j = i+1:nchan
        S1 = hilbert(EEG(i,:));
        S2 = hilbert(EEG(j,:));
        tS = S1.*conj(S2);
        isig = imag(tS);
        
        if strcmp(method,'PLI') == 1
            FC_matrix(i,j) = abs(mean(sign(isig)));
            FC_matrix(j,i) = FC_matrix(i,j);
        end

    end
end