# H4 feature-family diagnostic

This is a post-failure diagnostic on the frozen run and fold. The combined
HOG+HSV classifier remains the registered gate; weaker feature subsets
cannot be selected to pass.

| features | dimensions | AUC | bootstrap 95% CI |
|---|---:|---:|---:|
| hog+hsv | 1788 | 0.7964 | 0.7481–0.8392 |
| hog | 1764 | 0.7792 | 0.7321–0.8243 |
| hsv | 24 | 0.6816 | 0.6293–0.7292 |

Interpret the stronger subset as an engineering clue only. It does not
change the AUC 0.60 maximum or reopen M13.
