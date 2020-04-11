function Y = Shifting(F, dz)
[M,N] = size(F);
M1 = floor(M/2);
M2 = M1*2+M;
F1 = zeros(M2,N);
F1(M1+1:M1+M,:) = F;
w = (0:M2-1)*2*pi/M2;
W = repmat(w',[1,N]);
G = real(ifft(fft(F1).*exp(-1i*W*dz)));
Y = G(M1+1:M1+M,:);

% [~, Y] = meshgrid(1:N,1:M);
% Id = Y<=dz;
% G(Id) = 0;