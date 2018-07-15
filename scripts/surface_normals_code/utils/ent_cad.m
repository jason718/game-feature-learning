function[ent_val,max_pose,valid_models] = ent_cad(models,azang)
% compute the entropy value of given CAD models -- 
% azang is the azimuthal angle to bin the CAD models.
ed_angles = [10:azang:360];
mod_ang = (models(:,2)-floor(models(:,2)/36)*36)*10;
hist_ang = histc(mod_ang, ed_angles);
p_val = hist_ang/sum(hist_ang(:));

%keyboard;
[~,I] = max(hist_ang); max_pose = ed_angles(I);
valid_models = (mod_ang >= max_pose-10) & ...
	       (mod_ang <= max_pose+10);

ent_val = -sum(p_val(p_val~=0).*log2(p_val(p_val~=0)));

%if isempty(models)
%  pick = [];
%  return;
%end

% sort the values based on CAD score
%[~,I] = sort(models(:,3), 'descend');
%models = models(I,:);
%models(:,2) = (models(:,2) - ...
%	       floor(models(:,2)/36)*36)*10;
%cadInd = [1:length(models(:,1))]';
%
%pick = [];
%while(~isempty(cadInd))
	%
	%size(cadInd)
%	iter = cadInd(1);
%	ith_model_id = models(1,1);
%	ith_sim_cad = find(models(:,1) == ith_model_id);
	% find overlapping ones --
%	ith_thresh_cad = ith_sim_cad(min(abs(models(ith_sim_cad,2)-...
%				     models(1,2)), 360-abs(models(ith_sim_cad,2)-...
%                                     models(1,2)))<=azang);
%	pick = [pick; iter]; 
%	models(ith_thresh_cad,:) = [];
%	cadInd(ith_thresh_cad) = [];	
%end

end
