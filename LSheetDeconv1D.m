function F1 = LSheetDeconv1D(G, zPSF,N)
%% deconvolve along z direction pixel by pixel 
% G = flip(G,3);
[L1, L2, L3] = size(G);
F1 = zeros(size(G));
F2 = zeros(size(G));
%% 1D deconvolution by Wiener filtering
parfor j=1:L1
    convolved_image = squeeze(G(j,:,:))';
    zPSF1D = double(zPSF);
    J1 = zeros(size(convolved_image));
    J2 = J1;
    for i=1:size(convolved_image,2)
%         zPSF = LSheetPSF1(end-L3+1:end,:);
        I = double(convolved_image(:,i));
        J1(:,i) = deconvwnr(I,(zPSF1D(:,i)),5E-3);
    end
    F1(j,:,:) = J1';
%     F2(j,:,:) = J2';
%     imagesc(J1)
end    
% figure;imagesc(squeeze(F1(L1/2,:,:))');
return;
%% 1D deconvolution by Richardson-Lucy algorithm
P1 = repmat(zPSF,[1 1 L1]);
P2 = permute(P1, [3 2 1]);
zPSF1 = P2./repmat(sum(P2,3),[1 1 L3]);

zOTF = fftshift(fft(zPSF1,[],3),3);

X = G;
for i=1:N
    FX = fftshift(fft(X,[],3),3);
    T = G./real(fftshift(ifft(ifftshift(FX.*zOTF,3),[],3),3));
    FT = fftshift(fft(T,[],3),3);
    X = X.*real(fftshift(ifft(ifftshift(conj(zOTF).*FT,3),[],3),3));
    fprintf(strcat('Iteration number: i=', num2str(i),'\n'));
end
F1 = X;