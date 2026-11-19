# Unified Vietnamese OCR dataset

Package provider-trusted originals from VietHTR V1 (60,047 pairs) and compose
7–10 or 11–13 original lines from the same split. The exporter preserves text
and crop IDs in placement order, keeps original image bytes, and uses hardlinks
when available. Treat linked files as immutable. A portable package contains
all referenced images and the three split JSONLs. Known document and writer
IDs must stay in one split. Missing IDs limit isolation to available evidence.

```powershell
python scripts/export_unified_ocr.py --config configs/unified_ocr_profiles.yaml --dry-run
```

For a small smoke build, point the YAML output to a new directory under
outputs and pass `--smoke`. The full build and model training are deferred
until mentor approval.

The exported sampling configuration selects a profile explicitly. Full uses
20/40/30/10 for single lines, 7–10 lines, 11–13 lines and legal pages.
Viet-only uses 25/45/30/0. The Trainer samples with replacement on the training
split using its configured seed. Evaluation uses the existing loader.
Weighted sampling currently requires a single Trainer process.

Set `quotas` independently for train, validation and test. The example requests
4,000 and 3,000 train composites. Smoke caps each quota at one and uses at
most 40 originals per split. Crops may recur across distinct composites within
their split. Heuristic crop warnings travel with the ordered line metadata.
Unreadable or technically unusable lines are excluded from composition.
Canvas area is configurable and does not impose a model's pixel limit.
Dry-run estimates feasibility from metadata. The build reports actual
shortfalls after checking geometry.

Optional legal inputs are `legal_pages` (document bundles) and
`legal_curated` (hash-bound review export). The gate requires complete,
confirmed pages, matching render bytes and DOM transcripts, and no unresolved
warnings beyond the standard review reminder. The page transcript covers
ordinary visible text, with declared stamps and signatures excluded.
Structured field GT remains separate. Existing dev/test legal provenance
cannot supply the full profile's train quota.

Focused CPU fixtures cover both composite ranges, exact ordered GT, placement,
split provenance, legal provisional exclusion, dry-run and sampling weights.
