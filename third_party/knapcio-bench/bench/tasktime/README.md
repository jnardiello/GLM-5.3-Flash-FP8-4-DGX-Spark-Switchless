# tasktime: how long does an agent need to finish a small task?

Six seeded repositories (logic bugs, a date bug, a JS unicode/slug bug, a quadratic function to make linear,
a CLI feature with tests, a lossy SQLite migration). `run_tasktime.sh <launcher> <label>` copies each task,
lets the OpenCode launcher (`dscode` for DeepSeek-V4.1-Flash, `glmcode4` for GLM-5.3-Flash) work on it
non-interactively with a 15-minute cap, then verifies with the task's own tests and records wall seconds,
pass/fail and the number of tool lines. Compare the two `results/<label>/summary.tsv` files. Greedy
sampling is not guaranteed through the launchers; run each label twice.
