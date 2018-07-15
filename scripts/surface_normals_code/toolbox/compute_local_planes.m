% Computes local surface normal information. Note that in this file, the Y
% coordinate points up, consistent with the image coordinate frame.
%
% Args:
%   X - Nx1 column vector of 3D point cloud X-coordinates
%   Y - Nx1 column vector of 3D point cloud Y-coordinates
%   Z - Nx1 column vector of 3D point cloud Z-coordinates.
%   params
%
% Returns:
%   imgPlanes - an 'image' of the plane parameters for each pixel. 
%   imgNormals - HxWx3 matrix of surface normals at each pixel.
%   imgConfs - HxW image of confidences.
function [imgPlanes, imgNormals, imgConfs, N] = ...
    compute_local_planes(X, Y, Z, sz)

  blockWidths = [1 3 6 9];
  relDepthThresh = 0.05;

%  [~, sz] = get_projection_mask();
  H = sz(1);
  W = sz(2);

  N = H * W;

  pts = [X(:) Y(:) Z(:) ones(N, 1)];

  [u, v] = meshgrid(1:W, 1:H);

  blockWidths = [-blockWidths 0 blockWidths];
  [nu, nv] = meshgrid(blockWidths, blockWidths);

  nx = zeros(H, W);
  ny = zeros(H, W);
  nz = zeros(H, W);
  nd = zeros(H, W);
  imgConfs = zeros(H, W);

  ind_all = find(Z);
  for k = ind_all(:)'        

    u2 = u(k)+nu;
    v2 = v(k)+nv;

    % Check that u2 and v2 are in image.
    valid = (u2 > 0) & (v2 > 0) & (u2 <= W) & (v2 <= H);
    u2 = u2(valid);
    v2 = v2(valid);
    ind2 = v2 + (u2-1)*H;

    % Check that depth difference is not too large.
    valid = abs(Z(ind2) - Z(k)) < Z(k) * relDepthThresh;
    u2 = u2(valid);
    v2 = v2(valid);
    ind2 = v2 + (u2-1)*H;

    if numel(u2) < 3
      continue;
    end

    A = pts(ind2, :);        
    [eigv, l] = eig(A'*A);
    nx(k) = eigv(1,1);
    ny(k) = eigv(2,1);
    nz(k) = eigv(3,1);
    nd(k) = eigv(4,1);
    imgConfs(k) = 1 - sqrt(l(1) / l(2,2)); 
  end

  %make all the vectors point at the camera
  flip = reshape(sign(X(:).*nx(:)+Y(:).*ny(:)+Z(:).*nz(:)+eps),size(nx));
  N = bsxfun(@times,flip,cat(3,nx,ny,nz));
  %woo hoo graphics conventions
  N = bsxfun(@rdivide,N,sum(N.^2,3).^0.5+eps);

  % Normalize so that first three coordinates form a unit normal vector and
  % the largest normal component is positive
  imgPlanes = cat(3, nx, ny, nz, nd); % ./ repmat(len, [1 1 4]);
  len = sqrt(nx.^2 + ny.^2 + nz.^2);
  imgPlanes = imgPlanes ./ repmat(len+eps, [1 1 4]);
  imgNormals = imgPlanes(:, :, 1:3);
end
