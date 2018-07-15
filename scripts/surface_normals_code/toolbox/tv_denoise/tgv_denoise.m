function u = tgv_denoise(f, lambda, alpha0, alpha1, maxIterations, checkIterations)
%tgv_denoise Total generalized denoising via primal-dual algorithm
% The optimization solves the problem
%   arg min_u,uu { lambda/2 * |w*(u-f)|_2^2  +  alpha_1 |grad_u - uu|_2 + alpha_0 * |grad_uu|_2 }
% = arg min_u,uu { lambda/2 * |w*(u-f)|_2^2  +  max_pu,puu { <pu, grad_u - uu> + <puu, grad_uu> }} % primal
%                s.t. |pu|_2 < alpha_1, |puu|_2 < alpha_0
% = arg min_u,uu { lambda/2 * |w*(u-f)|_2^2  +  max_pu,puu { -div_pu * u - <pu, uu> - div_puu * uu } % dual
%                s.t. |pu|_2 < alpha_1, |puu|_2 < alpha_0
% I.e. gradient update steps for the dual variables are computed via the primal formulation, and their 
% constraints are handled via appropriate projection steps.
% Update step for the primal variables are taken from the dual formulation.
% 
% The L2 squared data term can be solved closed from via the prox step (u + tau*w*f)./(1 + tau*w).

alpha0 = alpha0/lambda;
alpha1 = alpha1/lambda;

% update step sizes
tau = 0.05;
L2 = 12; % lipschitz constant
sigma = 1/tau/L2;


% initilization of mask for invalid input pixels
w = isfinite(f);
f(~w) = 0;

% result initialized with input data term
u = f;

[M, N] = size(u);
% u_prev = u;

% vector fields
uu = zeros(M,N,2);
uu_prev = zeros(M,N,2);

% dual variables
pu = zeros(M,N,2);
puu = zeros(M,N,3);

% gradient
grad_u = zeros(M,N,2);
grad_uu = zeros(M,N,3);

% divergence
div_pu = zeros(M,N);
div_puu = zeros(M,N,2);

% energy
primalEnergy = [];

% iterate
for k=1:maxIterations
    %
    % primal update
    %
    
    u_prev = u;
    uu_prev = uu;
    
    div_pu = dxm(pu(:,:,1)) + dym(pu(:,:,2));
    div_puu(:,:,1) = dxm(puu(:,:,1)) + dym(puu(:,:,3));
    div_puu(:,:,2) = dxm(puu(:,:,3)) + dym(puu(:,:,2)); 

    u = u + tau.*div_pu;
    uu = uu + tau.*(pu+div_puu);
    
    % prox operator
    u = (u + tau*w.*f)./(1.0 + tau.*w);
    
    % over relaxation
    uu_prev = 2*uu - uu_prev;
    u_prev = 2*u - u_prev;
    
    
    %
    % dual update
    %
    
    % update dual pu
    grad_u(:,:,1) = dxp(u_prev);
    grad_u(:,:,2) = dyp(u_prev);
    pu = pu + sigma .* (grad_u - uu_prev);
    reproject = max(1.0, sqrt(sum(pu.^2,3)) ./ alpha1);
    pu = pu ./ repmat(reproject, [1,1,2]);
    
    % update dual puu
    grad_uu(:,:,1) = dxp(uu_prev(:,:,1));
    grad_uu(:,:,2) = dyp(uu_prev(:,:,2));
    grad_uu(:,:,3) = 0.5 * (dyp(uu_prev(:,:,1)) + dxp(uu_prev(:,:,2)));
    puu = puu + sigma.*grad_uu;
    reproject = max(1.0, sqrt(puu(:,:,1).^2 + puu(:,:,2).^2 + 2*puu(:,:,3).^2) ./ alpha0);
    puu = puu ./ repmat(reproject, [1,1,3]);
    
    
    %
    % energies
    %
    funcPrimal = alpha1 .* sqrt(sum((grad_u - uu_prev).^2,3)) + ...
        alpha0 .* sqrt(grad_uu(:,:,1).^2 + grad_uu(:,:,2).^2 + 2*grad_uu(:,:,3).^2) + ...
        (w.*(u-f)).^2 / 2.0;
    primalEnergy = [primalEnergy, sum(funcPrimal(:))];
    
    
    if mod(k,checkIterations) == 0
%         subplot(1,2,1), imshow(u,[]); drawnow;
%         subplot(1,2,2), plot(primalEnergy, 'k-'); drawnow;
        fprintf('TGV2-L2-PD: it = %4d, ep = %f\n', k, primalEnergy(end));
        %
        % optional: we could end when the energy does not change a lot
        %
        change = 1e-3;
        diff = abs(primalEnergy(end)-primalEnergy(end-1));
        if  diff < change
            break;
        end
    end
    
    
end

end

%% gradient calculations
function [dx] = dxm(u)
[M, N] = size(u);
dx = [u(:,1:end-1) zeros(M,1)] - [zeros(M,1) u(:,1:end-1)];
end

function [dy] = dym(u)
[M, N] = size(u);
dy = [u(1:end-1,:);zeros(1,N)] - [zeros(1,N);u(1:end-1,:)];
end

function [dx] = dxp(u)
dx = [u(:,2:end) u(:,end)] - u;
end

function [dy] = dyp(u)
dy = [u(2:end,:); u(end,:)] - u;
end


