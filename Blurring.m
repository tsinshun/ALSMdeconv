function Blur = Blurring(Img, OTF, P)
[L1, L2, L3] = size(Img);
Blur = zeros(L1,L2,L3);

% I = fftshift(ifftn(ifftshift(fftshift(fftn(Img.*P2)).*OTF)));

parfor i=1:L3
    P0 = Shifting(P, i-1);
    P1 = repmat(P0(end-L3+1:end,:),[1 1 L1]);
    P2 = permute(P1, [3 2 1]);
%     imagesc(squeeze(P2(1,:,:))');
    I = fftshift(ifftn(ifftshift(fftshift(fftn(Img.*P2)).*OTF)));
    Blur(:,:,i) = I(:,:,i);
end
