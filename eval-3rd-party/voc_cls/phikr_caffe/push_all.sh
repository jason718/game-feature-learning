while read a; do
  if [[ $a == "BRANCHES"* ]]; then
    BRANCHES=(${a:10:-1});
  fi
done < make_future.sh

BRANCH=$(git rev-parse --abbrev-ref HEAD)
for b in ${BRANCHES[@]}; do
  eval $(echo git checkout $b)
  git push -f
  if [[ $? != 0 ]]; then
    echo "Failed to push"
    exit 1
  fi
done
git checkout master
git push origin master

git checkout $BRANCH
