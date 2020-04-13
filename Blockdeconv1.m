function FF = Blockdeconv1(G, PSF, P2, B, N, lambda)
%%
FF = zeros(size(G));

[L1,L2,L3] = size(G);
M = 64;
% B = [500 750; 750 1250; 1250 1750];
Nb = length(B);
FB = []; FP = []; Xid =[]; k = 1;
for i=1:Nb
    x1 = B(i,1)-M;
    x2 = B(i,2)+M;
    
    if x1<1
       x1 = 1; 
    end
    
    if x2>L2
        x2 = L2;
    end
    
    if mod(abs(x2-x1+1 - size(PSF,2)),2)~=0 % mod(x2-x1+1)~=0 
        x2 = x2 -1;
    end
            
    Block = G(:,x1:x2,:);
%     Block = edgeTapering(G(:,x1:x2,:),PSF(:,:,L3/2));

    
    Lb = (size(PSF,2) - (x2-x1+1))/2;
    LsPSF = PSF(:,Lb+1:end-Lb,:).*P2(:,x1:x2,:);  
%     Lx = size(Block,2); x3 = 1024;
%     LsPSF = PSF(:,Lb+1:end-Lb,:).*P2(:,x3-Lx/2+1:x3+Lx/2,:);
    LsPSF = flip(LsPSF,3); % flip the PSF if necessary
        
%     F{k} = RichdLucy1(denoising(Block), lsPSF1, N,lambda); %1e-6
    Xid(k,:) = [abs(x1-B(i,1))+1 abs(x2-B(i,2))];
    
    FB{k} = Block; FP{k} = LsPSF;
    k = k + 1;
end

% FF = FP; return;

for k = 1:Nb
    
    if k<4
%         continue;
    end
   FB{k} = RichdLucy1GPU((FB{k}), FP{k}, N,lambda); %1e-6
%    FB{k} = deconvwnr(edgeTapering(FB{k},fspecial('gaussian',60,2)),FP{k},1e-7);
%     FB{k} = deconvlucy((FB{k}),FP{k},100);

end
  
for k=1:Nb
    x1 = Xid(k,1);
    x2 = Xid(k,2);
    F = FB{k};
    F(F<0) = 0;
    FF(:,B(k,1):B(k,2),:) = F(:,x1:end-x2,:);
end
 figure;imagesc(squeeze(FF(L1/2,:,:))'); colormap hot
 return;
 
 
 
