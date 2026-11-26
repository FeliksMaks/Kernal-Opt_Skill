# Correctness gate

A speed number counts only for a build whose output is checked. The check that worked: teacher-force a reference
model's greedy tokens through the engine and compare the engine's top-1 token at every position.

1. **Reference.** Run llama.cpp's `llama-server` on the same model file. `scripts/make_reference.py --server URL
   --prompts prompts.json --n 48 --out ref.json` stores, per prompt, the prompt ids, the greedy ids and the top-5
   log-probabilities of each step. Use chat-templated prompts (a fact, code, a story); never gate on a bare BOS prompt,
   where the top token is a near tie and any rounding flips it.
2. **Engine mode.** The engine needs a mode that takes ids and prints, for each position, its top tokens and the
   log-probability of the reference's next token. `scripts/score.py` reads lines of the form
   `S <pos> <next_id> <logprob_of_next> <rank_of_next> <id>:<lp> ...`; adapt the parser if the engine prints otherwise.
3. **Both paths.** Score the single-token path (feed one forced token at a time) and the prompt path (the whole sequence
   as one prompt). They are different kernels.
4. **Pass.** At least 98% agreement, and every miss a near tie: the reference's own gap between its top two tokens, or
   the engine's gap between its pick and the reference's token, below 0.5. A later build must not score lower.
5. **When it is close.** Short references (a few hundred positions) cannot separate two builds near 98%. Make a longer
   one: 30 or more prompts of 1-3K tokens, about 3,000 scored positions. Compare the mean KL divergence of both builds
   to an independent path of the same engine (for example its unfused kernel).

Logit dumps of long prompts are large (gigabytes per prompt for a 200K vocabulary). Delete each after scoring.
