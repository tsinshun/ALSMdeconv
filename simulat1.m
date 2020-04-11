load simul.mat % data for the input are required, which is can be accessed by this share link: https://1drv.ms/u/s!AkeSfNrAbS03eliG14zzh1T2HoE?e=79Krrp;

focus = 242; 


L1 = 256; L2 = 512; L3 = 128;
% Apod = repmat(squeeze(PSF(L1/2, L2/2, :)),[1,size(LSheetPSF,2)]);
imagesc(LSheetPSF1(end-L3+1:end,:));

% PSF
PSF0 = zeros(L1,L2*4,L3);
PSF0(:,L2*2-L2/2+1:L2*2+L2/2,:) = PSF;
OTF = fftshift(fftn(PSF0));

% use microtubule as object
Img = zeros(L1,L2*4,L3);
Img(:,1:L2,:) = Microtub;
Img(:,L2+1:L2*2,:) = Microtub;
Img(:,2*L2+1:L2*3,:) = Microtub;
Img(:,L2*3+1:L2*4,:) = Microtub;

Img = [Microtub Microtub fliplr([Microtub Microtub])];
%% simulate blurring
tic
Blur = BlurringGPU(Img, OTF, LSheetPSF1); 
toc
figure;imagesc(squeeze(Blur(L1/2,:,:))');

%% add Poisson noisy
G = poissrnd(mat2gray(Blur)*2e3 + 100);
% figure;imagesc(squeeze(G(L1/2,:,:))');
%% 1D deconvolution
Pattern = LSheetPSF1;
Apod=squeeze(PSF(256/2,512/2,:));
Apod=repmat(Apod,[1 2048]);
zPSF = Pattern(65:end-64,:).*Apod;
%%
F1 = LSheetDeconv1D(Blur, flipud(zPSF),10);
figure;imagesc(squeeze(max(F1,[],1))'); colormap hot%axis image
%% 3D deconvolution block by block
P1 = repmat(LSheetPSF1(64+1:end-64,:),[1 1 L1]);
P2 = permute(P1, [3 2 1]);
% P2 = flip(P2,3);
%% 
i = (0:7)';  B = [256*i+1 256*(i+1)];
FF = Blockdeconv1(G, PSF, P2, B,300,5E-4);

;



