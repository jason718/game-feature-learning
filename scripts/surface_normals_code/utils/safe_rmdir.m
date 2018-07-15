function safe_rmdir(fname)
try
  unix(sprintf('rm -rf %s', fname));
catch 
end
