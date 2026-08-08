# COVAS-VAD method specification

## 1. Input unit

COVAS-VAD scores fixed-duration clips generated from each original video.
For video metadata `(fps, frame_count)`, center frames are

```text
c_i = i * frame_interval, i = 0, 1, ...
```

until `c_i < frame_count`. The temporal window is

```text
start_i = max(0, c_i / fps - window_seconds / 2)
end_i   = min(frame_count / fps, c_i / fps + window_seconds / 2)
```

The defaults are `frame_interval=16`, `window_seconds=10`, `sample_fps=2`,
and `max_frames=10`. Neighboring clips overlap for normal surveillance frame
rates.

## 2. System prompt

The exact default system prompt is:

```text
You are a careful video anomaly detector for public surveillance scenes.
Judge only visible evidence in the current video segment. Use an ordered
anomaly-severity scale from 0 to 1: 0 means ordinary safe activity; 0.5
means clearly concerning or plausibly anomalous activity; and 1 means
unmistakably severe, dangerous, violent, criminal, or emergency activity.
Intermediate thresholds preserve this order. Do not infer events that are
not visually supported.
```

For threshold `tau`, the user question is:

```text
On the severity scale defined by the system, is the anomaly severity visible
in this video segment at least {tau}? Judge only the current video's visual
evidence. Answer exactly YES or NO.
```

Thresholds are `0.1, 0.2, ..., 1.0`.

## 3. Cumulative threshold likelihood

For each threshold, COVAS-VAD reads the next-token logits for the single-token
candidates `YES` and `NO`. With likelihood temperature `T`,

```text
p_tau = softmax([logit(NO), logit(YES)] / T)[YES].
```

The optimized path verifies that both candidates are exactly one token. The
reference path supports complete multi-token candidate likelihoods, but the
published optimized E0 configuration uses the single-token logits.

## 4. Shared visual computation

The ten threshold conversations refer to the same clip. The optimized path:

1. decodes and preprocesses the video once;
2. runs the vision encoder and multimodal projector once;
3. constructs the ten text suffixes against the shared visual features;
4. batches suffixes with equal length;
5. optionally evaluates their common multimodal/text prefix once and expands
   its KV cache across the threshold batch.

`threshold_batch_size` controls peak language-model activation memory without
changing the score definition.

## 5. Monotonic projection

Valid cumulative tail probabilities should satisfy

```text
P(S >= 0.1) >= ... >= P(S >= 1.0).
```

Finite-model estimates can violate this ordering. COVAS-VAD uses the
pool-adjacent-violators algorithm to compute the L2 projection onto the
non-increasing cone. This introduces no learned parameter.

## 6. Continuous score

The tail integral identity is

```text
E[S | V] = integral_0^1 P(S >= tau | V) d tau.
```

On the 0.1 grid, COVAS-VAD uses the right Riemann approximation

```text
score(V) = 0.1 * sum_{k=1}^{10} p_k
         = mean(p_1, ..., p_10).
```

This produces a continuous score despite using ten ordinal questions.

## 7. Error handling and checkpoints

Score JSON files are atomically replaced after a configurable number of newly
completed clips. A processing exception assigns `default_score` (0.5 by
default) and records the exception under `_errors/<video>.json`. On resume,
successful clips are skipped and failed clips are retried.

## 8. Evaluation

The E0 evaluation protocol:

1. numerically sort score JSON keys;
2. apply Gaussian smoothing to the clip-score sequence (`sigma=10`);
3. repeat each clip score for `frame_interval=16` frames;
4. crop or zero-pad to the annotated video length;
5. concatenate all test videos;
6. compute frame-level ROC-AUC, PR-AUC, and maximum F1.

All compared methods must use the same video list, score expansion, smoothing,
and temporal ground truth.

