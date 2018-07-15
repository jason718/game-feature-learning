function alreadyLocked = isLocked(fname)

if fileExists(fname) || mymkdir_dist([fname '.lock'])==0
  alreadyLocked = 1;
else
  alreadyLocked = 0;
end
