function [TV,R] = Roughness(X,lambda)

[L1, L2, L3] = size(X);

X(X<1)=1; 

if gpuDeviceCount>0
    gx = gpuArray(zeros(L1, L2, L3)); %gpuArray
    gy = gpuArray(zeros(L1, L2, L3));
    gz = gpuArray(zeros(L1, L2, L3));
else 
    gx = zeros(L1, L2, L3); %gpuArray
    gy = zeros(L1, L2, L3);
    gz = zeros(L1, L2, L3);
end

% gx = zeros(L1, L2, L3); %gpuArray
% gy = zeros(L1, L2, L3);
% gz = zeros(L1, L2, L3);

gy(1:L1-1,:,:) = diff(X,1,1);
gx(:,1:L2-1,:) = diff(X,1,2);
gz(:,:,1:L3-1) = diff(X,1,3);

[Ry, dRy] = getRough(X,gy,1);
[Rx, dRx] = getRough(X,gx,2);
[Rz, dRz] = getRough(X,gz,3);
if nargout>1
    R = Rx + Ry + Rz;
end
dR = dRx + dRy + dRz;%/5
dR(isnan(dR)) = 0;

TV = 1./(1+lambda*dR);

function [R, dR] = getRough(X,g,n)
if gpuDeviceCount>0
    T1 =  gpuArray(zeros(size(X)));
else
    T1 = zeros(size(X));
end
T = g./X;
T(isnan(T))=0;

R = g.*T;
switch n
    case 1  % y-direction
        T1(2:end,:,:) = 2*T(1:end-1,:,:);
    case 2  % x-direction
        T1(:,2:end,:) = 2*T(:,1:end-1,:);
    case 3  % z-direction
        T1(:,:,2:end) = 2*T(:,:,1:end-1);
        
end

dR = -2*T - T.*T + T1; 


