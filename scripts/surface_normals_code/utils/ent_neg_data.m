function[valid_models] = ent_neg_data(models, max_pose)
% compute the entropy value of given CAD models -- 
% azang is the azimuthal angle to bin the CAD models.
mod_ang = (models(:,2)-floor(models(:,2)/36)*36)*10;

valid_models = (mod_ang < max_pose-20) | ...
	       (mod_ang > max_pose+20);

end
