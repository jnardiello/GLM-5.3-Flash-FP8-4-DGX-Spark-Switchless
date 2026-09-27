**Verdict: E29 differs from the vendor FP8 model, but shows no detectable quality loss on this corpus.** R0 is the vendor FP8 weights served on the same four nodes with an FP8 KV cache. E29's next-token distributions do not match R0, so it fails the pre-registered "negligible deviation" test by a wide margin. On the actual next token, E29's perplexity change against R0 is indistinguishable from zero in the dense, sparse and all-position aggregates. It is also indistinguishable from zero in every corpus category over all positions. This is not proof of equivalent answer quality: the task comparison is underpowered, and the long-context evidence rests on few sources. NVFP4 deviates from R0 more than E29 does and is worse overall, beyond 2,048 tokens, on agentic code and on Italian. On short contexts it scores slightly better.

1. **Different: yes (pre-registered test: exceeds).**
   - In the dense regime (at most 2,048 conditioning tokens), Cm against R0 gives a mean coarse KL of 0.184 [0.165, 0.204] nats and 83.4% top-1 agreement.
   - The margins were a mean excess below 0.002 nats, a p99 excess below 0.02 nats and a top-1 drop below 0.5 pp. They assumed that small numeric changes produce small deviations.
   - The dense-regime recipe comparisons measured here mostly land near the same value:
     - Cpre against R0: 0.182.
     - Cm against Cpre: 0.182.
     - The September 19 base recipe against R0: 0.173.
     - The E21 step: 0.172.
     - The E03 mHC step alone is smaller, at 0.075. The runtime-only steps after E22b are near zero.

   For scale, R0 against itself beyond 2,048 tokens gives 0.277 (sparse regime, from engine nondeterminism). A dense KL of this size reflects disagreement between numerically different recipes; on its own it does not show a quality loss.
2. **Worse: not detected.** E29's perplexity change against R0 on the actual next token:
   - dense: +0.03% [−0.56, +0.60];
   - sparse: +0.19% [−0.34, +0.73];
   - all positions: +0.14% [−0.32, +0.56].

   R0 against a second run of itself gives +0.21% [−0.09, +0.48] over all positions. Every corpus category over all positions also includes zero, for example agentic code +0.35% [−0.39, +1.11] and Italian chat −0.04% [−0.19, +0.12].

   Two dense-regime category intervals exclude zero, both in E29's favour: synthetic Italian chat −0.31% [−0.57, −0.02] and long context −2.86% [−4.99, −0.22]. The long-context category has only five windows.
3. **Sparse regime (beyond 2,048 tokens): unresolved.** The engine is not deterministic there: R0 against itself gives 0.277 nats and 80.0% top-1 agreement. Three executions per arm were collected, but no validated estimator exists for fidelity between population-average distributions when execution variability differs between arms. Descriptively, Cm (0.294) sits close to that floor: its single-execution excess is 0.017 nats, with a UB95 of 0.019.
4. **Where the difference comes from (ladder, 128 dense windows).** Each step's perplexity change is measured against the previous rung:
   - R0 → September 19 base recipe (hybrid KDA): +1.74% [+0.94, +2.65].
   - September 19 base → Cpre (E03 mHC): KL 0.075, +0.09% [−0.54, +0.62]. Cumulative against R0: +1.83% [+0.72, +2.78].
   - Cpre → LE21 (E21 projections): −1.82% [−2.80, −0.96]. Cumulative against R0: −0.02% [−1.16, +0.96].

   The runtime steps from E22b to E29 (drafter, scheduler and KV pool) leave the measured dense-regime logprobs bit-identical or nearly so, in measurement mode:
   - LE21 → LE22b is bit-identical (0 of 231,195 rows differ).
   - LE22b → E29 differs in 5 rows of one window: mean KL 8 × 10⁻⁸ nats, one top-1 flip.
5. **Tasks: no detected difference, but the tasks cannot establish equivalence.**
   - On qeval greedy (75 mixed tasks: 25 code, 16 reasoning, 14 maths, 8 JSON, 7 formatting, 5 prose), R0 passes 73/75 and E29 73/75, with 2 discordant tasks in each direction (McNemar p = 1.00). NVFP4 passes 74/75.
   - The ±2 pp equivalence criterion needs far more tasks. The observed discordance (4 of 75 pairs) gives an approximate paired MDE of 7.5 pp (normal approximation to McNemar).
   - The cloud (Z) arm was not run, so the task criterion remains unresolved.

**Is E29 more or less precise than NVFP4? More precise.**

- **Distance from R0 (dense, one execution):**
  - NVFP4: 0.346 [0.319, 0.373] nats, 75.1% top-1 agreement, about 1.9 times E29's coarse KL.
  - E29: 0.184 nats, 83.4%.

  NVFP4 is not deterministic even in the dense regime: 155,206 of 155,390 rows differ between two runs. On the three-execution subset, the KL between run-averaged distributions is 0.229 for NVFP4 against R0 and 0.169 for E29 (descriptive).
- **Quality, NVFP4 against E29:** perplexity changes on the actual next token:
  - sparse: +11.8% [+7.9, +15.5];
  - all positions: +6.9% [+4.5, +9.2];
  - agentic code: +8.9% [+6.3, +11.7];
  - synthetic Italian chat: +10.1% [+9.3, +10.9].

  In the dense regime alone, NVFP4 scores −3.4% [−5.4, −1.3] against E29 and −3.3% [−5.4, −1.2] against R0. That short-context advantage is present in this data but unexplained, and it does not carry over to longer contexts.
- **Corruption probe** (not run on R0):
  - Italian replies containing U+FFFD replacement characters: NVFP4 3 of 40 (8 characters in total), E29 0 of 40.
  - Tool calls parsed correctly: 10 of 10 on both.
- **Tasks:** 74/75 against 73/75, no detectable difference at this power.

**Limitations.** These limitations apply to the results above:
- **Coarse KL is a lower bound.** On the dense rows of the 85-window K = 100 subset, Cm against R0 gives 0.273 nats at K = 100 and 0.194 for the same rows truncated to K = 20.
- **R0 shares the FP8-KV error.** It uses the FP8 KV cache, because BF16 KV is not available on B12X.
- **Scope of the distribution results.** They come from cold, serial measurement mode; the production bridge was not measured.
- **Corpus coverage.** The corpus is a single provisional manifest that the owner has not frozen. The Italian conversations are synthetic, and long context rests on five windows.
- **Deferred work.** The voxel showcase and the cloud arm are deferred as next steps.
