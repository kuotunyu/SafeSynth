# SafeSynth Evidence-First Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stock Gradio demo with the approved `zh-TW` evidence-first research interface while preserving model behavior and scientific claims.

**Architecture:** Keep inference in `app.py` and drawing in `src/inference/demo.py`, add a small `src/inference/demo_ui.py` view-model layer for testable HTML, and place responsive visual rules in `assets/demo_ui.css`. A curated CC0 example is inferred at startup and feeds a native Gradio `ImageSlider`; image upload and reset callbacks return the same immutable presentation structure.

**Tech Stack:** Python 3.12, Gradio 6.22, NumPy, Pillow, OpenCV, pytest, Ruff, CSS.

## Global Constraints

- Primary language is Traditional Chinese (`zh-TW`); technical terms remain in original English where translation reduces clarity.
- Body copy is at least `18px`; auxiliary and technical copy is at least `16px`.
- Keep `RT-DETRv2-R18`, `real_only`, and frozen `threshold=0.07` unchanged.
- Do not change dataset, model weights, evaluation results, or latency claims.
- Do not add a JavaScript build step, frontend framework, network font, decorative gradient, or fabricated evidence.
- Compliance state must use text in addition to color.
- Git author and committer must remain `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`.

---

## File Map

- Create `assets/demo/example.jpg`: curated CC0 example copied from the released dataset.
- Create `assets/demo_ui.css`: tokens, layout, typography, focus, responsive and reduced-motion rules.
- Create `src/inference/demo_ui.py`: immutable presentation model and escaped `zh-TW` HTML formatters.
- Create `tests/test_demo_ui.py`: formatter and presentation tests.
- Modify `src/inference/demo.py`: accessible localized box captions and Morandi-aligned colors.
- Modify `tests/test_demo.py`: pin localized caption semantics and drawing behavior.
- Modify `app.py`: default sample resolution, presentation callbacks, Gradio composition and CSS loading.
- Modify `README.md`: update the live-demo description and screenshot/GIF context if its behavior changed materially.

### Task 1: Localized and accessible detection rendering

**Files:**
- Modify: `src/inference/demo.py`
- Modify: `tests/test_demo.py`

**Interfaces:**
- Consumes: existing `ComplianceStatus`, `DrawnBox`, `drawn_boxes()`, and `draw_on()`.
- Produces: `DrawnBox.semantic_label: str` and localized `caption` text used by the live image and video renderers.

- [ ] **Step 1: Write failing semantic-label tests**

Add assertions that `helmet`, `head`, and `person` render `已佩戴`, `未佩戴`, and `僅定位`, while confidence remains visible when width permits.

```python
def test_captions_carry_a_zh_tw_semantic_label() -> None:
    boxes = drawn_boxes(
        [_detection(0, 0.9), _detection(1, 0.8), _detection(2, 0.7)],
        class_names=CLASSES,
        score_threshold=0.0,
    )
    captions = {box.label: box.caption for box in boxes}
    assert "已佩戴" in captions["helmet"]
    assert "未佩戴" in captions["head"]
    assert "僅定位" in captions["person"]
```

- [ ] **Step 2: Run the targeted test and confirm RED**

Run: `uv run pytest tests/test_demo.py -q`

Expected: failure because current captions use English verdict values and `person` has no semantic text.

- [ ] **Step 3: Implement semantic labels and palette constants**

Use BGR constants that render the approved RGB colors after `[::-1]`:

```python
COMPLIANT_COLOUR = (138, 157, 127)      # RGB #7F9D8A after reversal
NON_COMPLIANT_COLOUR = (114, 125, 195)  # RGB #C37D72 after reversal
NEUTRAL_COLOUR = (154, 145, 127)        # RGB #7F919A after reversal

SEMANTIC_LABEL = {
    ComplianceStatus.COMPLIANT: "已佩戴",
    ComplianceStatus.NON_COMPLIANT: "未佩戴",
    None: "僅定位",
}
```

The full caption is `<class> · <semantic label> · <score>`. Width degradation keeps semantic meaning before dropping to the raw class.

- [ ] **Step 4: Run focused rendering tests**

Run: `uv run pytest tests/test_demo.py -q`

Expected: all demo rendering tests pass.

- [ ] **Step 5: Commit the rendering unit**

```powershell
git add src/inference/demo.py tests/test_demo.py
git commit -m "feat(demo): localize compliance rendering"
```

### Task 2: Testable UI presentation model and curated example

**Files:**
- Create: `src/inference/demo_ui.py`
- Create: `tests/test_demo_ui.py`
- Create: `assets/demo/example.jpg`

**Interfaces:**
- Consumes: `FrameSummary` and measured inference timings.
- Produces: `ImagePresentation`, `format_summary_html()`, `format_evidence_html()`, `format_error_html()`, and `load_example_image()`.

- [ ] **Step 1: Copy the approved CC0 example asset**

Copy `D:\sdg-data\02-safesynth\raw\hard-hat-detection\images\hard_hat_workers863.png` to `assets/demo/example.jpg`, converting to RGB JPEG quality 92. Record the source filename in `demo_ui.py` as `DEFAULT_EXAMPLE_SOURCE`.

- [ ] **Step 2: Write failing presentation tests**

Create tests for a normal result and an indeterminate frame:

```python
def test_summary_html_leads_with_plain_language_counts() -> None:
    html = format_summary_html(FrameSummary(4, 3, 1), source_label="精選範例")
    assert "7 位人員中" in html
    assert "4 位正確佩戴安全帽" in html
    assert "57" in html
    assert "精選範例" in html


def test_indeterminate_summary_never_claims_zero_percent() -> None:
    html = format_summary_html(FrameSummary(0, 0, 2), source_label="使用者影像")
    assert "未找到可判定" in html
    assert "0%" not in html
```

Also test HTML escaping by passing a source label containing `<script>`.

- [ ] **Step 3: Run the new tests and confirm RED**

Run: `uv run pytest tests/test_demo_ui.py -q`

Expected: import failure because `src.inference.demo_ui` does not exist.

- [ ] **Step 4: Implement the presentation module**

```python
@dataclass(frozen=True)
class ImagePresentation:
    comparison: tuple[np.ndarray, np.ndarray]
    summary_html: str
    evidence_html: str
    source_html: str
    error_html: str = ""


def format_error_html(message: str) -> str:
    return f'<div class="ss-error" role="alert">{escape(message)}</div>'


def load_example_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
```

`format_summary_html()` branches on `summary.compliance_rate is None`, escapes the source label, and emits the approved count sentence, rate, tiles, and legend. `format_evidence_html()` escapes model metadata and emits model-only, end-to-end, checkpoint, threshold, and device values. Both return semantic markup with `aria-live="polite"` on changing result status.

- [ ] **Step 5: Run module tests and Ruff**

Run: `uv run pytest tests/test_demo_ui.py -q`

Run: `uv run ruff check src/inference/demo_ui.py tests/test_demo_ui.py`

Expected: PASS.

- [ ] **Step 6: Commit the presentation unit**

```powershell
git add assets/demo/example.jpg src/inference/demo_ui.py tests/test_demo_ui.py
git commit -m "feat(demo): add curated evidence presentation"
```

### Task 3: Image callbacks and evidence-first Gradio composition

**Files:**
- Modify: `app.py`
- Modify: `tests/test_demo_ui.py`
- Create: `assets/demo_ui.css`

**Interfaces:**
- Consumes: `ImagePresentation` formatters and existing `annotate()`.
- Produces: `present_image(detector, image, threshold, source_label) -> ImagePresentation`, `load_uploaded_image(path) -> np.ndarray`, and the redesigned `build_interface()`.

- [ ] **Step 1: Write failing callback tests with a fake detector**

```python
def test_present_image_returns_before_after_and_real_metrics() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = app.present_image(_FakeDetector(), image, 0.07, source_label="使用者影像")
    assert result.comparison[0] is image
    assert result.comparison[1].shape == image.shape
    assert "使用者影像" in result.summary_html
    assert "end-to-end" in result.evidence_html
```

Add tests for unreadable paths and a fake detector exception. The error result must preserve the original image in both slider positions.

- [ ] **Step 2: Run callback tests and confirm RED**

Run: `uv run pytest tests/test_demo_ui.py -q`

Expected: failure because `present_image()` and `load_uploaded_image()` do not exist.

- [ ] **Step 3: Implement callback helpers**

`load_uploaded_image()` accepts `str | Path`, opens through Pillow, applies EXIF transpose, converts to RGB, and raises `DemoInputError` with the approved `zh-TW` message for invalid files.

`present_image()` calls `annotate()`, formats the summary and measured evidence, and catches inference errors only at the presentation boundary so model internals remain testable.

- [ ] **Step 4: Build the Gradio hierarchy**

Use:

- first `gr.HTML` containing the five-block direction contract and compact product bar;
- `gr.Tab("圖片偵測")` and `gr.Tab("影片偵測")`;
- `gr.Row(elem_classes="evidence-hero")`;
- `gr.ImageSlider` at scale 8;
- summary `gr.HTML` and primary `gr.UploadButton` at scale 4;
- `gr.Button("回到精選範例")` as the secondary action;
- evidence strip `gr.HTML`;
- closed `gr.Accordion("研究方法與限制", open=False)`.

The startup value comes from running `present_image()` on `assets/demo/example.jpg`. Upload and reset events return the same four presentation outputs.

- [ ] **Step 5: Implement the CSS file**

Define exact tokens from the spec and style only scoped Gradio classes. Required rules include:

```css
:root {
  --ss-canvas: #e7e3da;
  --ss-paper: #f5f2eb;
  --ss-ink: #17201e;
  --ss-yellow: #e1b45b;
  --ss-sage: #7f9d8a;
  --ss-coral: #c37d72;
  --ss-graphite: #151b19;
}
.gradio-container { font-size: 18px; }
.ss-meta, .ss-legend { font-size: 16px; }
button, [role="button"] { min-height: 44px; }
@media (max-width: 780px) { .evidence-hero { flex-direction: column; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
```

- [ ] **Step 6: Load CSS through `launch(css_paths=PROJECT_ROOT / "assets" / "demo_ui.css")`**

Pass `PROJECT_ROOT / "assets" / "demo_ui.css"` from `main()` and set footer links to the GitHub, Dataset, and Model URLs rather than the default Gradio promotional footer.

- [ ] **Step 7: Run callback and existing demo tests**

Run: `uv run pytest tests/test_demo_ui.py tests/test_demo.py tests/test_demo_gif.py -q`

Run: `uv run ruff check app.py src/inference/demo.py src/inference/demo_ui.py tests/test_demo.py tests/test_demo_ui.py`

Expected: PASS.

- [ ] **Step 8: Commit the image experience**

```powershell
git add app.py assets/demo_ui.css tests/test_demo_ui.py
git commit -m "feat(demo): build evidence-first image experience"
```

### Task 4: Video progress, localized states, and disclosure content

**Files:**
- Modify: `app.py`
- Modify: `tests/test_demo.py`
- Modify: `tests/test_demo_ui.py`

**Interfaces:**
- Consumes: existing `annotate_video()` and `VideoResult`.
- Produces: optional `progress_callback(current: int, total: int | None)`, localized `video_summary_html()`, and user-recoverable error states.

- [ ] **Step 1: Write failing progress and copy tests**

Extend `annotate_video()` tests with a callback that records `(current, total)` and assert that updates are monotonic and end at the decoded frame count. Assert that undecodable video renders `無法讀取影片 frames`.

- [ ] **Step 2: Run targeted tests and confirm RED**

Run: `uv run pytest tests/test_demo.py tests/test_demo_ui.py -q`

Expected: failure because the callback and localized formatter are absent.

- [ ] **Step 3: Add progress without changing frame limits**

Add:

```python
def annotate_video(
    detector,
    source,
    threshold: float,
    destination: Path | None = None,
    *,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> VideoResult | None:
```

Read `CAP_PROP_FRAME_COUNT` when positive, cap the reported total at `MAX_VIDEO_FRAMES`, and call the callback after every appended frame.

- [ ] **Step 4: Connect Gradio progress and localized output**

The video callback accepts `progress=gr.Progress()` and maps progress updates to `progress(current / total, desc="正在分析影片…")`. It returns structured HTML rather than the old English textbox note.

- [ ] **Step 5: Verify video regressions**

Run: `uv run pytest tests/test_demo.py tests/test_demo_ui.py -q`

Expected: PASS including truncation, median latency, mean compliance rate, and progress.

- [ ] **Step 6: Commit the video experience**

```powershell
git add app.py tests/test_demo.py tests/test_demo_ui.py
git commit -m "feat(demo): clarify video progress and recovery"
```

### Task 5: Browser finish, documentation, and release-quality verification

**Files:**
- Modify: `README.md` only if the launch copy or screenshots require correction.
- Create at finish: `DESIGN.md` and Impeccable sidecar if the shipped documenter requires it.

**Interfaces:**
- Consumes: finished local demo, approved mockup, and direction contract.
- Produces: verified Desktop/Mobile experience and recorded design system.

- [ ] **Step 1: Start the redesigned demo on CPU**

Run:

```powershell
uv run python app.py --weights "D:\sdg-data\02-safesynth\publish\safesynth-rtdetrv2-r18" --device cpu --port 7870
```

Use another unoccupied port if 7870 is unavailable; do not terminate unrelated processes.

- [ ] **Step 2: Capture the first inspection batch**

Capture Desktop `1440×1000` and Mobile `390×844` for:

- default example;
- custom upload success;
- Image／Video tabs;
- opened research disclosure.

Check font sizes, first-viewport hierarchy, horizontal overflow, focus visibility, touch targets, and console errors in one batch.

- [ ] **Step 3: Apply one batched correction pass**

Fix every material problem from the first screenshots together. Do not enter an open-ended pixel-polishing loop.

- [ ] **Step 4: Capture the confirmation batch**

Recapture Desktop and Mobile at the same viewports and compare against the approved evidence-stage mockup.

- [ ] **Step 5: Run the mechanical detector once**

Run:

```powershell
node C:\Users\3Hml\.codex\skills\impeccable\scripts\detect.mjs --json app.py
```

Record any false positives and fix mechanical findings in one batch.

- [ ] **Step 6: Run complete project verification**

```powershell
uv run ruff check .
uv run pytest -q
uv lock --check
uv run python -m scripts.verify_readme
uv run python -m scripts.check_commit_identity
git diff --check
```

Expected: all checks pass; the contributor identity check reports only `kuotunyu`.

- [ ] **Step 7: Record the shipped visual system**

Write `DESIGN.md` from the actual rendered UI, including tokens, typography, component vocabulary, responsive behavior, and accessibility rules. The document describes what shipped rather than the pre-build intention.

- [ ] **Step 8: Commit final documentation and verification fixes**

```powershell
git add README.md DESIGN.md app.py assets src tests
git diff --cached --check
git commit -m "docs(demo): record evidence-first design system"
```

## Plan Self-Review

- Spec coverage: every approved requirement maps to Tasks 1–5.
- Placeholder scan: complete; no deferred implementation markers remain.
- Type consistency: `ImagePresentation`, `present_image()`, and all HTML formatter names are defined once and consumed consistently.
- Scope: one Gradio surface, its focused presentation module, rendering semantics, tests, assets, and documentation; no unrelated training or evaluation changes.
- Execution choice: user explicitly delegated implementation, so execute inline with `superpowers:executing-plans` without another approval interruption.
