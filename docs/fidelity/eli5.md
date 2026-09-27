We asked a simple question: does our tuned E29 setup of GLM-5.3-Flash predict text as well as the official FP8 model it is built from? Both run on the same four machines with the same 8-bit memory cache.

- **How we checked.** We gave both models the same 2.5 million tokens of text: coding-agent sessions, synthetic Italian chats and model-written answers. At every token we asked each model what it expected next, and how surprised it was by the token that actually came next.
- **They don't always pick the same favourite token.** On short contexts, E29's first choice matches the official model 83% of the time. Several other recipe changes we measured shift it about as much, the way two equally strong chess engines can pick different good moves.
- **We found no quality loss.** E29's surprise (perplexity) differs by +0.1%. The 95% range (−0.3% to +0.6%) includes zero, and so do the ranges for code, Italian and long contexts. On 75 mixed tasks both models pass 73. That shows no measurable loss on this test set, not proof that every answer is as good.
- **A public 4-bit version (NVFP4) loses quality on longer text.** It is about 7% more surprised by real text overall, 12% beyond 2,000 tokens of context and 10% in Italian, although about 3% less surprised on short contexts. It wrote broken characters in 3 of 40 Italian answers; E29 wrote none.
- **One detail stays open.** Beyond 2,000 tokens the serving engine gives slightly different predictions from run to run, even for the official model. There we can't measure the fine-grained distance, only the overall quality, which shows no detectable change.

**Bottom line:** E29 is not a bit-for-bit copy of the official model, but we could not measure any quality loss.
