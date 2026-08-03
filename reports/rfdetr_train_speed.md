# Measured training speed on this machine

- Device: `NVIDIA GeForce RTX 4090`
- Arm: `real_only`, seed `1337`
- Method: two runs (40 and 140 steps); the rate is the SLOPE between them, so fixed setup cost cancels rather than being amortised into a per-step number.
- Warmup steps forced to 1 for the probe. The production schedule warms up over 2,000, which would otherwise be the only thing a short probe measures.

| Model | it/s | ms/step | fixed cost (s) | 10900 steps |
|---|---:|---:|---:|---:|
| `rfdetr` | 2.89 | 346 | 10.5 | **1.05 h** |

Reference point, measured not assumed: the four production arms ran on a Colab L4 at 1.7-1.9 it/s, about 1.6-1.75 hours each.

A negative fixed cost means the two runs disagree about setup overhead; the safety gate rejects that probe before it can write an ETA report.
