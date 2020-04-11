function X = RichdLucy1(G, PSF, N,lambda, Init)
%% function to do 3D deconvolution by Richardson-Lucy algorithm with TV regularization
% Input: 
%   G: 3D blurred noisy image, 
%   PSF: 3D PSF
%   N: iteration number
%   Init: Initial guest(optional)
% Output: 
%   X: final deconvolved 3D image
% reference: 
%   [1] Dey.Nicolas, et al. "Richardson¨CLucy algorithm with total variation regularization for 3D confocal microscope deconvolution." Microscopy research and technique 69.4 (2006): 260-266.
%   [2] Sage, Daniel, et al. "DeconvolutionLab2: An open-source software for deconvolution microscopy." Methods 115 (2017): 28-41.
% Author: Shun Qin
% Contact: shun.qin@phys.uni-goettingen.de


if nargin<4
   lambda = 0.001; 
end

if nargin>=5 && ~isempty(Init)
   X = Init;
else
   X = G;
end

% RichdLucy2(G, PSF, N,lambda, X); return;

PSF = PSF/sum(PSF(:));
OTF = fftshift(fftn(PSF,size(G)));%
OTF1 = conj(OTF);

for i=1:N
    X(X<1)=1; %X(X>2^16)=2^16;
    TV = Roughness(X, lambda);
%     TV = 1;
%     TV = getTV(X, lambda); % isotropy TV

    T = G./real(fftshift(ifftn(ifftshift(fftshift(fftn(X)).*OTF))));   
    X = X.*TV.*real(fftshift(ifftn(ifftshift(fftshift(fftn(T)).*OTF1)))); % flip(x) => conj(X) flip(P)?

    if sum(isnan(X(:)))>0
        error(['NAN found in iteration: ' num2str(i)]);
    end
    fprintf(strcat('nIter = ', num2str(i), '\n'));
%     imagesc(squeeze(X(size(G,1)/2,:,:))'); colormap gray
end

X(X<0)=0;
% figure;imagesc(squeeze(X(size(G,1)/2,:,:))'); colormap gray
return;


function TV = getTV(X,lambda)

[L1, L2, L3] = size(X);

if gpuDeviceCount>0
    gx = gpuArray(zeros(L1, L2, L3)); %gpuArray
    gy = gpuArray(zeros(L1, L2, L3));
    gz = gpuArray(zeros(L1, L2, L3));
else 
    gx = zeros(L1, L2, L3); %gpuArray
    gy = zeros(L1, L2, L3);
    gz = zeros(L1, L2, L3);
end


gy(1:L1-1,:,:) = -diff(X,1,1);
gx(:,1:L2-1,:) = -diff(X,1,2);
gz(:,:,1:L3-1) = -diff(X,1,3);

% gy(2:L1,:,:) = flip(diff(flip(X,1),1,1),1);
% gx(:,2:L2,:) = flip(diff(flip(X,2),1,2),2);
% gz(:,:,2:L3) = flip(diff(flip(X,3),1,3),3);

[gx, gy, gz] = gradNormalize(gx, gy, gz);

if gpuDeviceCount>0
    ggx = gpuArray(zeros(L1, L2, L3));
    ggy = gpuArray(zeros(L1, L2, L3));
    ggz = gpuArray(zeros(L1, L2, L3));
else 
    ggx = zeros(L1, L2, L3);
    ggy = zeros(L1, L2, L3);
    ggz = zeros(L1, L2, L3);
end



% ggy(1:L1-1,:,:) = -diff(gy,1,1);
% ggx(:,1:L2-1,:) = -diff(gx,1,2);
% ggz(:,:,1:L3-1) = -diff(gz,1,3);

ggy(2:L1,:,:) = diff(gy,1,1);
ggx(:,2:L2,:) = diff(gx,1,2);
ggz(:,:,2:L3) = diff(gz,1,3);


TV = 1./((ggx+ggy+ggz)*lambda + 1);

function [gx, gy, gz] = gradNormalize(gx, gy, gz)
norm = sqrt(gx.^2 + gy.^2 + gz.^2);
gx = gx./norm;
gy = gy./norm;
gz = gz./norm;

gx(isnan(gx))=0;
gy(isnan(gy))=0;
gz(isnan(gz))=0;


