#!/bin/zsh
# Time-to-green for each seeded task with an OpenCode launcher. usage: ./run_tasktime.sh <launcher: dscode|glmcode4> <label>
set -u
L=$1; LABEL=$2; HERE=${0:A:h}; OUT=$HERE/results/$LABEL; mkdir -p $OUT; : > $OUT/summary.tsv
for T in $HERE/tasks/*/; do
  N=$(basename $T); W=$OUT/$N; rm -rf $W; cp -r $T $W; cd $W; git init -q; git add -A; git -c user.name=t -c user.email=t@t commit -q -m seed
  t0=$(date +%s.%N)
  perl -e "alarm 900; exec @ARGV" $L run "$(cat TASK.md) Work only inside this directory. Run the tests yourself and stop when they pass." > $W/agent.log 2>&1; rc=$?
  t1=$(date +%s.%N)
  if [ -f package.json ]; then node --test > $W/verify.log 2>&1; ok=$?; else perl -e "alarm 120; exec @ARGV" uv run -q --with pytest python -m pytest -q -x -p no:cacheprovider > $W/verify.log 2>&1; ok=$?; fi
  secs=$(printf '%.0f' $(echo "$t1 - $t0" | bc)); turns=$(grep -c -i -E "tool|bash|edit|write" $W/agent.log)
  printf "%s\t%s\t%s\t%s\t%s\n" "$N" "$secs" "$( [ $ok = 0 ] && echo PASS || echo FAIL)" "$rc" "$turns" | tee -a $OUT/summary.tsv
done
echo "task	secs	verify	agent_rc	tool_lines" | cat - $OUT/summary.tsv
