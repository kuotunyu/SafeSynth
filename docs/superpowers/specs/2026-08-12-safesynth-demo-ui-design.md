# SafeSynth Demo UI/UX Redesign

**Date:** 2026-08-12

**Status:** Approved for automatic implementation

**Surface:** `app.py` Gradio demo

**Primary audience:** GitHub visitors evaluating the project

## Objective

Replace the stock Gradio prototype presentation with a polished evidence-first research demo. A first-time visitor must understand within five seconds that SafeSynth performs hard-hat compliance detection, see a real result, and know how to try another image. The redesign must improve presentation without changing model behavior, evaluation claims, or the frozen operating point.

## Approved Direction

The approved direction is **B — 證據優先展示台**.

- The annotated result is the visual center of the first viewport.
- A compact right summary explains the verdict in plain `zh-TW`.
- Research evidence supports the result below the primary interaction.
- Detailed methodology, negative results, calibration notes, and limitations use progressive disclosure.
- The visual language is deliberate and restrained: warm paper surfaces, graphite image stage, safety yellow action, sage compliance, dusty coral non-compliance, and slate neutral detections.

## Typography and Density

- Body text: `18px` minimum.
- Secondary labels and technical metadata: `16px` minimum.
- Main result heading: `32–40px` responsive.
- Compliance rate: `72–88px` responsive.
- Navigation and buttons: `17px` minimum.
- Desktop content width: at most `1480px`, with `16–24px` internal gaps.
- Empty decoration, repeated headings, duplicate labels, and low-value whitespace are prohibited.

## First Viewport

1. Compact top bar:
   - SafeSynth name and product mark.
   - `Research Demo` context label.
   - Model-ready status.
   - GitHub and Hugging Face links as secondary actions.
2. Image／Video segmented navigation.
3. Evidence stage:
   - Large Before／After `gr.ImageSlider` using a curated default example.
   - Right summary with a plain-language result sentence, compliance rate, two count tiles, and accessible legend.
   - One primary action: upload another site image.
4. Evidence strip:
   - Real measured end-to-end latency for the current inference.
   - `RT-DETRv2-R18` and `real_only` checkpoint.
   - Frozen `threshold=0.07`.
   - Links to reproducibility evidence.

## Default Example

- Ship `assets/demo/example.jpg`, copied from the CC0 release source `images/hard_hat_workers863.png`.
- The image was selected because it shows an active construction context, multiple visible helmets and bare heads, readable box sizes, and less distracting watermarking than the existing first GIF frame.
- Run the default image through the same `annotate()` path during interface construction; do not ship a manually fabricated output.
- The UI labels it `精選範例` so visitors do not mistake it for their upload.
- Provide `回到精選範例` after a custom upload.

## Image Interaction

- Interface startup displays the default example and its real model output.
- Upload accepts JPG, JPEG, PNG, and WEBP.
- During inference, disable conflicting actions and show `正在分析影像…`.
- On success, return:
  - `(original_image, annotated_image)` to the slider;
  - structured summary HTML;
  - performance HTML from the measured inference;
  - visible state `使用者影像`.
- If no helmeted or bare head can be judged, show `未找到可判定的安全帽佩戴狀態`; the compliance rate is `—`, never `0%`.
- Removing or clearing an upload restores the curated example rather than leaving a dead blank surface.

## Video Interaction

- Video remains a separate tab and does not compete with Image in the first viewport.
- Upload copy states the first-120-frames limit before processing begins.
- Use `gr.Progress` to report decoded frame progress where Gradio supports it.
- The result includes annotated video, frame count, whether truncation occurred, mean compliance rate when defined, and median latency.
- Invalid or undecodable video preserves the uploaded file and displays an actionable `zh-TW` message.

## Result Rendering and Accessibility

- `helmet`: sage green solid box with visible `已佩戴` text.
- `head`: dusty coral box with visible `未佩戴` text and a distinct line pattern or label treatment.
- `person`: slate box with visible `僅定位` text and no compliance verdict.
- Summary repeats semantic labels; color is never the only carrier of meaning.
- Keyboard focus must remain visible on tabs, upload, reset, links, and accordions.
- Touch targets must be at least `44px` in both dimensions.
- `prefers-reduced-motion` removes nonessential transitions.

## Research Disclosure

An accordion titled `研究方法與限制` contains:

- Why `threshold=0.07` is intentional and Validation-frozen.
- Why the demo serves `real_only`.
- The negative result: synthetic arms did not improve the main RT-DETRv2 detection result.
- `person` annotation limitations and why that class carries no verdict.
- Latency measurement conditions and the distinction between model-only and end-to-end time.

The accordion starts closed. Its wording must remain accurate to the README and evaluation artifacts.

## Visual System

| Token | Value | Use |
|---|---|---|
| Canvas | `#E7E3DA` | Page background |
| Paper | `#F5F2EB` | Primary surfaces |
| Ink | `#17201E` | Main text |
| Muted | `#5E6965` | Secondary text |
| Safety yellow | `#E1B45B` | Primary action and focus accent |
| Sage | `#7F9D8A` | Compliant state |
| Dusty coral | `#C37D72` | Non-compliant state |
| Slate | `#7F919A` | Neutral `person` state |
| Graphite | `#151B19` | Image evidence stage |

- No decorative gradients.
- No glassmorphism beyond an optional restrained image overlay.
- Border radius stays between `10px` and `18px`; avoid a page made entirely of floating cards.
- Use the platform font stack with `Microsoft JhengHei` and system sans-serif; do not add a network font dependency.

## Implementation Boundaries

- Keep model loading and `Detector` behavior unchanged.
- Extract view formatting into focused helpers rather than expanding `build_interface()` with repeated HTML assembly.
- Create `assets/demo_ui.css` for tokens, responsive layout, focus states, and Gradio overrides.
- Keep semantic result formatting unit-testable without launching Gradio.
- Do not add a frontend framework or JavaScript build step.
- Do not change dataset, model weights, evaluation config, or published scientific claims.

## Error States

| Condition | User-visible response |
|---|---|
| Unsupported or corrupt image | `無法讀取這個影像。請使用 JPG、PNG 或 WEBP。` |
| No judgeable head | `未找到可判定的安全帽佩戴狀態。你可以換一張人物頭部更清楚的影像。` |
| Inference exception | Preserve the input and show `分析失敗，請重新執行；若問題持續，請查看終端機紀錄。` |
| Undecodable video | Preserve the upload and show `無法讀取影片 frames。請使用一般 MP4 編碼後重試。` |
| Model unavailable at startup | Keep existing CLI error behavior; do not render a misleading ready UI. |

## Test Strategy

1. Unit tests:
   - default example resolution and presence;
   - structured summary for compliant, non-compliant, and indeterminate frames;
   - performance metadata and `zh-TW` copy;
   - reset-to-example behavior;
   - image callback success and error behavior using a fake detector;
   - video progress and error copy.
2. Existing regression suite:
   - `tests/test_demo.py`
   - `tests/test_demo_gif.py`
   - full `pytest` and `ruff`.
3. Browser verification:
   - Desktop approximately `1440×1000`.
   - Mobile approximately `390×844`.
   - Default example, upload success, indeterminate result, Image／Video navigation, and research accordion.
   - No horizontal overflow, clipped content, unreadably small text, or browser console errors.
4. Mechanical UI detector:
   - Run once after the finished UI is rendered.

## Completion Criteria

- The first viewport matches the approved evidence-first mockup in hierarchy and density.
- Body content is at least `18px`; auxiliary content is at least `16px`.
- A visitor sees a real inference result without uploading first.
- The user can upload another image, understand the result, and recover from errors.
- Desktop and Mobile screenshots pass a single design review and one confirmation pass.
- All tests, Ruff, README verification, and Git contributor-identity checks remain green.
