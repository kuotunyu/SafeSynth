# H4 controlled blending ablations

All rows use the same 300 sample ids, frozen background/scenario schedule,
per-sample RNG derivation, fixed RNG consumption in noise matching, identical
group-disjoint folds, and the same HOG+HSV C=1 classifier. This isolates the
named blending change instead of reshuffling later samples or post-effects.

| variant | AUC | decision |
|---|---:|---|
| Final baseline: decontamination on, alpha core 192, sigma 0.60+0.02×side clipped 0.60–2.00 | **0.7964** | retain |
| Edge decontamination off | 0.8216 | worse |
| Edge core alpha 128 | 0.8054 | worse |
| Edge core alpha 250 | 0.8237 | worse |
| No second erosion (bank alpha only) | 0.8551 | much worse |
| Three-level RGB multiband alpha blend | 0.8507 | much worse; reverted |
| Strong feather, sigma clipped 1.00–3.00 | 0.8068 | worse |
| Weak feather, sigma clipped 0.40–1.00 | 0.7906 | statistically negligible |
| Very weak feather, sigma clipped 0.20–0.60 | 0.7950 | statistically negligible |

The best observed change is only 0.0058 AUC below the retained baseline and is
far inside H4's bootstrap interval. Selecting it on the gate evaluation would
be test-set threshold chasing, not evidence that paste artifacts were fixed.
The pre-registered scale-up maximum remains 0.60 and M11 remains blocked.

The ablations do establish two useful facts:

1. source-background colour decontamination materially helps;
2. simply increasing feather strength or removing the second erosion makes the
   pasted-vs-real shortcut stronger;
3. multiband blending spreads a low-frequency patch signature and also makes
   the shortcut stronger.

One read-only diagnostic points to composition context: on the final H4
classifier scores, pasted headlike boxes anchored near a `person` have score-AUC
0.7562 against real controls, versus 0.8134 for unanchored boxes. This is not a
replacement gate result, but it is a stronger next-step clue than further edge
tuning. The next attempt should pre-register a context-anchored composition
method and evaluate it on a new frozen group fold.
