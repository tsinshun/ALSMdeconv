function Blur = BlurringGPU(Img, OTF, P)
Img = gpuArray(Img);
OTF = gpuArray(OTF);

[L1, L2, L3] = size(Img); Lp = size(P,1);
P3 = zeros(L1,L3,L2); Lz = L3;
for i=1:L3
%     P0 = Shifting(P, i-1);
%     P3(i,:,:) = P0(end-L3+1:end,:);
    
    z = Lp-i; 
    P3(i,:,:) = P(z-Lz+1:z,:);
end

Blur = gather(blurring(Img, OTF, gpuArray(P3)));

function Blur = blurring(Img, OTF, P3)
[L1, L2, L3] = size(Img);
Blur = gpuArray(zeros(L1,L2,L3));
for i=1:L3
    P0 = P3(i,:,:);
    P1 = repmat(P0,[L1 1 1]);
    P2 = permute(P1, [1 3 2]);
%     imagesc(squeeze(P2(1,:,:))');
    I = fftshift(ifftn(ifftshift(fftshift(fftn(Img.*P2)).*OTF)));
    Blur(:,:,i) = real(I(:,:,i));
end
