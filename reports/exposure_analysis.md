# The arms compared at equal annotation budget

[TRAIN-07](../docs/training_spec.md) equalises optimizer STEPS, which controls compute. It does not equalise how often each arm sees a real photograph: a batch drawn from a 50/50 pool carries half as many real images, so at the same step the synthetic arms have seen each real image half as many times. Annotation is the resource this project claims to save, so it is the axis the method has to be judged on.

⚠️ **Matching real exposure unmatches compute.** At one pass over the real training set a 50/50 arm has taken twice as many optimizer steps as a real-only arm. The honest description of every row below is *same labels, more compute* — which is exactly the trade synthetic data offers, but it is not *same conditions*.

![exposure curves](reports/figures/exposure_curves.png)

Re-aggregated from each arm's `trainer_state.json` (EVAL-12). **Validation only**, single seed.

## mAP 50-95 at matched real-image exposure

| passes over real set | `real_only` | `standard_aug` | `unfiltered_syn` | `filtered_syn` |
|---:|---:|---:|---:|---:|
| 2 | 0.1860 | 0.1657 | 0.2001 | 0.2764 |
| 3 | 0.2352 | 0.2461 | 0.2472 | 0.2736 |
| 4 | 0.2852 | 0.2754 | 0.2500 | 0.2968 |
| 5 | 0.3095 | 0.2661 | 0.2752 | 0.2826 |
| 6 | 0.3377 | 0.2643 | 0.2897 | 0.2455 |
| 7 | 0.3449 | 0.2753 | 0.2714 | 0.3194 |
| 8 | 0.3563 | 0.2697 | 0.2973 | 0.3009 |
| 9 | 0.3499 | 0.2637 | 0.2987 | 0.2936 |
| 10 | 0.3428 | 0.2904 | 0.2453 | 0.2618 |
| 11 | 0.3474 | 0.2659 | 0.2745 | 0.2808 |
| 12 | 0.3468 | 0.2236 | 0.2506 | 0.2768 |
| 13 | 0.3464 | 0.2792 | 0.2459 | 0.2463 |
| 14 | 0.3279 | 0.2564 | 0.2172 | 0.2375 |
| 15 | 0.3300 | 0.2972 | 0.2611 | 0.2247 |
| 16 | 0.3200 | 0.3276 | 0.2471 | 0.2685 |
| 17 | 0.3341 | 0.3040 | 0.2137 | 0.2304 |
| 18 | 0.3220 | 0.3093 | 0.2552 | 0.2399 |
| 19 | 0.3265 | 0.2432 | 0.2432 | 0.2045 |
| 20 | 0.3174 | 0.2523 | 0.2605 | 0.2186 |
| 21 | 0.3083 | 0.2815 | 0.2459 | 0.2192 |
| 22 | 0.2902 | 0.2450 | 0.2316 | 0.2210 |
| 23 | 0.3101 | 0.2574 | 0.2312 | 0.2188 |
| 24 | 0.3046 | 0.2490 | 0.2303 | 0.2063 |

| challenger | largest lead | at | verdict |
|---|---:|---:|---|
| `standard_aug` | +0.0108 | 3 | leads up to about 16 passes over the real set, then falls behind |
| `unfiltered_syn` | +0.0141 | 2 | leads up to about 3 passes over the real set, then falls behind |
| `filtered_syn` | +0.0903 | 2 | leads up to about 4 passes over the real set, then falls behind |

## AP_small at matched real-image exposure

| passes over real set | `real_only` | `standard_aug` | `unfiltered_syn` | `filtered_syn` |
|---:|---:|---:|---:|---:|
| 2 | 0.1376 | 0.1333 | 0.1332 | 0.2176 |
| 3 | 0.1805 | 0.1760 | 0.1867 | 0.2024 |
| 4 | 0.2253 | 0.2006 | 0.2067 | 0.2254 |
| 5 | 0.2538 | 0.1967 | 0.2136 | 0.2152 |
| 6 | 0.2820 | 0.1988 | 0.2263 | 0.1896 |
| 7 | 0.2916 | 0.2112 | 0.2245 | 0.2369 |
| 8 | 0.3042 | 0.2107 | 0.2342 | 0.2332 |
| 9 | 0.2947 | 0.2068 | 0.2353 | 0.2205 |
| 10 | 0.2765 | 0.2364 | 0.1906 | 0.1869 |
| 11 | 0.2989 | 0.2092 | 0.2167 | 0.1981 |
| 12 | 0.2899 | 0.1803 | 0.2003 | 0.1901 |
| 13 | 0.2924 | 0.2273 | 0.1935 | 0.1686 |
| 14 | 0.2782 | 0.2016 | 0.1758 | 0.1607 |
| 15 | 0.2683 | 0.2529 | 0.2052 | 0.1472 |
| 16 | 0.2683 | 0.2788 | 0.1972 | 0.1765 |
| 17 | 0.2871 | 0.2495 | 0.1671 | 0.1448 |
| 18 | 0.2708 | 0.2635 | 0.1977 | 0.1535 |
| 19 | 0.2768 | 0.2018 | 0.1876 | 0.1382 |
| 20 | 0.2697 | 0.2102 | 0.2021 | 0.1441 |
| 21 | 0.2621 | 0.2319 | 0.1947 | 0.1458 |
| 22 | 0.2477 | 0.1918 | 0.1812 | 0.1486 |
| 23 | 0.2658 | 0.2013 | 0.1817 | 0.1472 |
| 24 | 0.2646 | 0.2110 | 0.1788 | 0.1398 |

| challenger | largest lead | at | verdict |
|---|---:|---:|---|
| `standard_aug` | +0.0105 | 16 | leads up to about 16 passes over the real set, then falls behind |
| `unfiltered_syn` | +0.0063 | 3 | leads up to about 3 passes over the real set, then falls behind |
| `filtered_syn` | +0.0800 | 2 | leads up to about 4 passes over the real set, then falls behind |

## What this does and does not establish

It does not overturn the main table. At the full step budget, and at every arm's own best checkpoint, `real_only` is ahead — that result stands and is reported as it is.

What it adds is where the synthetic data was doing something. If a challenger's lead is large at one or two passes over the real set and gone by four or five, then the composites help while real labels are scarce and stop helping once they are not. This dataset supplies 5,000 labelled images, which is the regime where synthetic augmentation has least to offer.

**Single seed.** [EVAL-10](../docs/evaluation_spec.md) forbids claiming a win from a small single-seed gap, and early training is the noisiest part of the curve. Treat the crossover as a direction worth testing, not as a measured constant.

**The experiment this points at** is a real-data-fraction ablation: retrain on 10%, 25% and 50% of the real training set with and without the same synthetic pool. If the reading above is right, the gap should widen as the real fraction shrinks. That is a cheaper experiment than the one already run, because every arm in it trains on less data.
