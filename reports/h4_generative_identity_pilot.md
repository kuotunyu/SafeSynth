# H4 generative identity pilot

- Status: **failed human identity gate**
- Images: **64**
- Scenario: `context_replacement`
- Root seed: `20260727`
- H4 AUC computed: **no**
- Outside-edit changed pixels: **0**
- Protected-core changed pixels: **0**
- Pixel-invariant scope: immediate inpaint result before global postfx
- Human gate: 0 label mismatches and at most 3 severe identity failures.
- Human decision: **rejected by `kuotunyu` on 2026-07-27**.
- Observed severe identity failures: **more than 3** (the exact count was not
  requested or inferred).
- Reviewer-labelled cell 02 visually matches canonical contact-sheet cell 10
  (`s20260727_000010`). Its source background,
  `hard_hat_workers861.png`, already contains mirrored top and bottom border
  artifacts and should have been rejected before generative editing.
- Reviewer-labelled cell 04 visually matches canonical contact-sheet cell 12
  (`s20260727_000012`). Its replacement anchor is source annotation `11122`,
  box `[187, 385, 65, 30]`; it ends one pixel from the image bottom and targets
  a truncated or reflected border object rather than the central worker helmet.
- Label mismatches were not separately counted, because exceeding the severe
  identity-failure limit already fails the gate.
- Consequence: Option A does not proceed to the one-shot H4 artifact gate.
- Contact sheet: `reports/figures/h4_generative_identity_pilot.png`
- Detail contact sheet: `reports/figures/h4_generative_identity_pilot_detail.png`

Each numbered cell is ordered DRAFT / EDIT MASK / REFERENCE / OUTPUT.
The cyan rectangle marks the registered editable boundary band.
The detail sheet crops DRAFT, EDIT MASK, and OUTPUT around that band.

The automated `PASS` text in a cell refers only to deterministic compositor and
filter checks. It is not a human visual approval and did not override this
failed gate.
