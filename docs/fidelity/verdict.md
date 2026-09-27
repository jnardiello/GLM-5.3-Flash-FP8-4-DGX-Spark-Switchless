**Verdict: E29 is different from the vendor FP8 model, but not worse.** Its next-token distributions do not match R0, so it fails the pre-registered "negligible deviation" test by a wide margin. On real text it predicts the actual next token as well as R0, with no detectable perplexity change at any context length. NVFP4 deviates further from R0 than E29 does. It is measurably worse beyond 2,048 tokens, on agentic code and on Italian.

1. **Different: yes (pre-registered test: exceeds).** In the dense regime (at most 2,048 conditioning tokens), Cm against R0 gives a mean coarse KL of 0.184 [0.165, 0.204] nats and 83.4% top-1 agreement. The margins were a 0.002-nat mean excess and a top-1 drop below 0.5 pp. Those margins assumed that small numeric changes produce small deviations. On this model and engine, any change that is not bit-identical produces about the same dense KL:
   - Cpre against R0: 0.182.
   - Cm against Cpre: 0.182.
   - The single E03 mHC ladder step: 0.075.
   - R0 against itself beyond 2,048 tokens: 0.277.

   KL measures disagreement here, not loss of quality.
2. **Worse: no.** E29's perplexity change against R0 on the actual next token:
   - dense: +0.03% [−0.56, +0.60];
   - sparse: +0.19% [−0.34, +0.73];
   - all positions: +0.14% [−0.32, +0.56].

   Every interval includes zero. R0 against a second run of itself gives +0.21% [−0.09, +0.48]. Every corpus category also includes zero, for example agentic code +0.35% [−0.39, +1.11] and Italian chat −0.04% [−0.19, +0.12].
3. **Sparse regime (beyond 2,048 tokens): unresolved.** The engine is not deterministic there: R0 against itself gives 0.277 nats and 80.0% top-1 agreement. No validated estimator exists for an arm's excess over that floor. Descriptively, Cm (0.294) sits close to the floor: its single-execution excess is 0.017 nats with a UB95 of 0.019. The pre-registered verdict nevertheless stays unresolved.
4. **Where the difference comes from (ladder, 128 dense windows).** Perplexity changes against R0:
   - R0 → September 19 base recipe (hybrid KDA): +1.74% [+0.94, +2.65].
   - E03 mHC: KL 0.075, +0.09% [−0.54, +0.62].
   - E21 residual projections: −1.82% [−2.80, −0.96]. This returns the cumulative change to −0.02% [−1.16, +0.96].

   The runtime steps from E22b to E29 (drafter, scheduler and KV pool) do not change the output:
   - LE21 → LE22b is bit-identical (0 of 231,195 rows differ).
   - LE22b → E29 differs in 5 rows of one window: mean KL 8 × 10⁻⁸ nats, one top-1 flip.
5. **Tasks: no detected difference, but the tasks cannot establish equivalence.** On qeval greedy (75 deterministic tasks), R0 passes 73/75 and E29 73/75, with 2 discordant tasks in each direction (McNemar p = 1.00). NVFP4 passes 74/75. The ±2 pp equivalence criterion needs far more tasks, since the approximate MDE at n = 75 is 16 pp. The cloud (Z) arm was not run, so the task criterion remains unresolved.

**Is E29 more or less precise than NVFP4? More precise.**

- **Distance from R0 (dense):**
  - NVFP4: 0.346 [0.319, 0.373] nats, 75.1% top-1 agreement.
  - E29: 0.184 nats, 83.4%.

  NVFP4 is not deterministic even in the dense regime: 155,206 of 155,390 rows differ between two runs. On the three-execution subset, the KL between run-averaged distributions is 0.229 for NVFP4 against R0 and 0.169 for E29 (descriptive).
- **Quality, NVFP4 against E29:** perplexity changes on the actual next token:
  - sparse: +11.8% [+7.9, +15.5];
  - all positions: +6.9% [+4.5, +9.2];
  - agentic code: +8.9% [+6.3, +11.7];
  - Italian chat: +10.1% [+9.4, +10.9].

  In the dense regime alone NVFP4 scores −3.4% [−5.4, −1.3]. That short-context advantage is real in this data but unexplained, and it does not carry over to longer contexts.
- **Corruption probe:**
  - Italian replies containing U+FFFD replacement characters: NVFP4 3 of 40 (8 characters in total), E29 0 of 40.
  - Tool calls parsed correctly: 10 of 10 on both.
- **Tasks:** 74/75 against 73/75, no detectable difference at this power.

**Limitations.** These limitations apply to the results above:
- Coarse KL is a lower bound: the K = 100 subset gives 0.273 nats against 0.194 at K = 20.
- R0 uses the FP8 KV cache, because BF16 KV is not available on B12X, so it shares the FP8-KV error.
- The corpus is a single provisional manifest that the owner has not frozen.
- The voxel showcase and the cloud arm are deferred as next steps.
