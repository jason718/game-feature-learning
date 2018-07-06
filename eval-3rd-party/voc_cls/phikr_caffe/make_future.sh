git checkout master
git pull

BRANCHES=('make_future' 'resize' 'fast_hdf5' 'transform_layer' 'transforming_fast_hdf5' 'array' 'net_from_string' 'color_conv' 'wait_for_cuda' 'repack' 'python_util' 'openmp' 'import_fix')
echo "***************************************************************** "
echo "********************* Updating all branches ********************* "
echo "***************************************************************** "
git fetch origin
for b in ${BRANCHES[@]}; do
  git checkout $b
  git reset --hard origin/$b
  git rebase master
  if [[ $? != 0 ]]; then
    echo "Make future failed"
    exit 1
  fi
done

echo "***************************************************************** "
echo "********************** Creating the future ********************** "
echo "***************************************************************** "
# git branch -D future
git branch -D past
git branch -m future past
git checkout -b future master
for b in ${BRANCHES[@]}; do
  git merge $b --rerere-autoupdate --no-edit
  if [[ $? != 0 ]]; then
    git commit --no-edit
    if [[ $? != 0 ]]; then
      echo "Failed to make future"
      exit 1
    fi
  fi
done
# CuDNN v5
#hub merge https://github.com/BVLC/caffe/pull/3919 --rerere-autoupdate --no-edit
#if [[ $? != 0 ]]; then
#  git commit --no-edit
#  if [[ $? != 0 ]]; then
#    echo "Failed to make future"
#    exit 1
#  fi
#fi

git push --set-upstream private future -f
