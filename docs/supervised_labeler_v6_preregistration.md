# Supervised labeler v6 pre-registration

v5 R50 failed its frozen audit at the selected raw threshold. A diagnostic
using only the already-consumed v5 calibration and failed-audit images found
that one simple geometry filter corrected the precision/recall tradeoff:

- reject predicted boxes covering more than 8% of the image;
- reject predicted boxes taller than 35% of the image;
- select the score threshold at a calibration precision floor of 0.82.

The calibration-selected v5 candidate was threshold 0.021, with precision
0.8213 and recall 0.8199. Applying that identical candidate to the consumed
v5 failed audit gave precision 0.8191, recall 0.7739, and median IoU 0.8377.
Those values are diagnostic only and cannot pass a gate.

v6 freezes the same R50 revision, training seed, optimization, geometry rules,
and threshold-selection policy before selecting a new group-disjoint Train
audit. All v1-v5 consumed groups become calibration history. Validation and
Test remain unread, and the FLUX generation gate remains locked until v6
passes both the numeric audit and a subsequent human review.
