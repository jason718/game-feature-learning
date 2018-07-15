function success = waitTillExists(fileNames, pauseInterval)
% Wait for a file to be created before proceeding
%
% Author: saurabh.me@gmail.com (Saurabh Singh)
if nargin < 2
  pauseInterval = 20;
end

maxIter = 30;
success = true;
for i = 1 : length(fileNames)
  fileName = fileNames{i};
  iterNo = 1;
  while ~fileExists(fileName)
    sleepTime = floor(10 + pauseInterval * rand(1, 1));
    fprintf('Dint find file [%s][%d], will wait [%d]\n', fileName, ...
      iterNo, sleepTime);
    pause(sleepTime);
    iterNo = iterNo + 1;
    if iterNo > maxIter
      fprintf('Giving up on file [%s]\n', fileName);
      success = false;
      break;
    end
  end
  if ~success
    break;
  end
end
end