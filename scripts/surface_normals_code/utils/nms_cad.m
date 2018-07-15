function pick = nms_cad(models,azang)
% top = nms(boxes, overlap)

if isempty(models)
  pick = [];
  return;
end

% sort the values based on CAD score
[~,I] = sort(models(:,3), 'descend');
models = models(I,:);
models(:,2) = (models(:,2) - ...
	       floor(models(:,2)/36)*36)*10;
cadInd = [1:length(models(:,1))]';
%
pick = [];
while(~isempty(cadInd))
	%
	%size(cadInd)
	iter = cadInd(1);
	ith_model_id = models(1,1);
	ith_sim_cad = find(models(:,1) == ith_model_id);
	% find overlapping ones --
	ith_thresh_cad = ith_sim_cad(min(abs(models(ith_sim_cad,2)-...
				     models(1,2)), 360-abs(models(ith_sim_cad,2)-...
                                     models(1,2)))<=azang);
	pick = [pick; iter]; 
	models(ith_thresh_cad,:) = [];
	cadInd(ith_thresh_cad) = [];	
end

end
