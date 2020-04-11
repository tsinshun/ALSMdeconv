function X = RichdLucy1GPU(G, PSF, N, lambda,Init)
G = gpuArray(G);
PSF = gpuArray(PSF);
if nargin>4
%    Init = gpuArray(Init); 
   X = gather(RichdLucy1(G, PSF, N, lambda,Init));
else
   X = gather(RichdLucy1(G, PSF, N,lambda));
end

