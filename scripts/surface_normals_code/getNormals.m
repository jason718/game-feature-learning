% this script helps in getting the surface normals for
% the videos of NYU depth dataset--
clc; clear all;

% addpath to the required 
addpath(genpath('./toolbox/'));
addpath('./utils/');

% dumping place
CACHEDIR = ['./vids_normal/'];
if(~isdir(CACHEDIR))
	mkdir(CACHEDIR);
end

% read the set of scenes to be considered
vid_set = load('vid_trainval_scenes.mat', 'img_set');
vid_set = vid_set.img_set;

% get the projection mask for NYUD2
[projectionMask, projectionSize] = get_projection_mask();

for i = 1:length(vid_set)


	% ith video scene
	ith_vid = vid_set{i};

	% make the directory for that particular scene
	SEQ_DIR = [CACHEDIR, ith_vid, '/'];
	if(~isdir(SEQ_DIR))
		mkdir(SEQ_DIR);
	end
	
	% get the list of frames for this scene category -- 
	frameList = get_synched_frames(['./nyu_depth_v2_raw/', ith_vid]);

	for j = 1:length(frameList)
	
		% save-file-name
		save_file_name = [SEQ_DIR, num2str(j, '%06d')];
		if(exist([save_file_name, '.mat'], 'file'));
			continue;
		end

		if(isLocked(save_file_name))
			continue;
		end
		
		%
		display([SEQ_DIR, num2str(j, '%06d'), '/', num2str(length(frameList), '%06d')]);


		% read the jth-depth image --
		imgDepth = imread(['./nyu_depth_v2_raw/', ith_vid, '/',...
				    frameList(j).rawDepthFilename]);
		imgDepth = swapbytes(imgDepth);
		imgRgb = imread(['./nyu_depth_v2_raw/', ith_vid, '/',...
                                     frameList(j).rawRgbFilename]);		

		% generally rgbImages and depthMap are not aligned. So one needs to 
		% first align them before use.
		[imgDepth2, imgRgb2] = project_depth_map(imgDepth, imgRgb);	
			
		% once the points are aligned (say according to RGB perspective)
		% compute 3D points.
		points3d = rgb_plane2rgb_world(imgDepth2);
		points3d = points3d(projectionMask,:);

	        % using 3D points, compute surface normals --
       		X = points3d(:,1);
        	Y = -points3d(:,2);
        	Z = points3d(:,3);
        	[imgPlanes, imgNormals, normalConf,NCompute] = ...
              		compute_local_planes(X, Y, Z, projectionSize);

	        NMask = sum(NCompute.^2,3).^0.5 > 0.5;

	        % tv-denoise the surface normals 
		% this is same as David Fouhey's stuff.
      		Ndash  = tvNormal(NCompute,1);
 	    	N1 = zeros(size(imgRgb2,1), size(imgRgb2,2));
		N1(projectionMask) = Ndash(:,:,1);

    	    	N2 = zeros(size(imgRgb2,1), size(imgRgb2,2));
        	N2(projectionMask) = Ndash(:,:,2);

        	N3 = zeros(size(imgRgb2,1), size(imgRgb2,2));
        	N3(projectionMask) = Ndash(:,:,3);
	
		N = cat(3, N1, N2, N3);
		nx = N(:,:,1); ny = N(:,:,2); nz = N(:,:,3);
        	Nn = (nx.^2 + ny.^2 + nz.^2).^0.5 + eps;
        	nx = nx./Nn; ny = ny./Nn; nz = nz./Nn;

        	% valid depth data --
        	depthValid = zeros(size(imgRgb2,1), size(imgRgb2,2));
        	depthValid(projectionMask) = NMask;

        	save([save_file_name, '.mat'], 'nx', 'ny', 'nz',...
                                'depthValid', 'imgRgb2', '-v7.3');

        	% create visualization --
        	Nvis =  uint8(255*(max(min(cat(3,nx,ny,nz),1),-1)+1)/2);
        	Nx = Nvis(:,:,1); Nx(~depthValid) = 0;
        	Ny = Nvis(:,:,2); Ny(~depthValid) = 0;
        	Nz = Nvis(:,:,3); Nz(~depthValid) = 0;
        	Nvis = uint8(cat(3,Nx, Ny, Nz));
		
		imwrite(imgRgb2, [save_file_name, '_rgb.png']);
        	imwrite(Nvis, [save_file_name, '_norm.png']);
		
		 % unlock the file 
	        unlock(save_file_name);

	end

end

