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
   pasted-vs-real shortcut stronger.

The next attempt should change the method (for example context-aware source
selection or a luminance-only multiband blend) and be evaluated on a new,
pre-frozen group fold rather than tuning these parameters further.
