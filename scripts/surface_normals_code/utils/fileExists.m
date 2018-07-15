function res = fileExists(filename)
% Author: saurabh.me@gmail.com (Saurabh Singh).
fid = fopen(filename,'r');
if fid == -1
  res = false;
else
  fclose(fid);
  res = true;
end
end
