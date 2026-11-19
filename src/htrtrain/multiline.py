"""Build small, explicit line-composite datasets without changing source data."""
from __future__ import annotations
import hashlib
import json
from itertools import islice
from pathlib import Path
import yaml
from PIL import Image, ImageDraw, ImageOps
from .dataset import DatasetError, iter_split
from .crop_quality import analyze

def normalize_line(image, settings):
    """Estimate foreground height and resize uniformly; keep source pixels around ink."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    diagnostic, _ = analyze(image)
    if diagnostic['severity'] == 'reject':
        raise DatasetError('invalid source image: ' + ', '.join(diagnostic['reasons']))
    gray = ImageOps.grayscale(image)
    hist = gray.histogram()
    # Otsu separates dark strokes from the local paper distribution.
    total = sum(hist)
    weighted = sum(i * value for i, value in enumerate(hist))
    left_count = left_sum = 0
    best, threshold = -1.0, 0
    for level, value in enumerate(hist):
        left_count += value
        left_sum += level * value
        right_count = total - left_count
        if left_count and right_count:
            score = left_count * right_count * (left_sum / left_count - (weighted - left_sum) / right_count) ** 2
            if score > best:
                best, threshold = score, level
    mask = gray.point(lambda p: 255 if p <= threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise DatasetError("no detectable text")
    x1, y1, x2, y2 = bbox
    ink_height = y2 - y1
    if ink_height < settings['min_source_text_height']:
        raise DatasetError("source text height too small")
    scale = settings['target_text_height'] / ink_height
    if scale > settings['max_scale'] or scale < settings.get('min_scale', 0.7):
        raise DatasetError("extreme scale ratio")
    # Never remove source pixels; clipping and neighboring lines need review.
    warnings = []
    if x1 <= 1 or y1 <= 1 or x2 >= image.width - 1 or y2 >= image.height - 1:
        warnings.append('ink_near_source_edge_possible_clipping')
    if ink_height / image.height > 0.88:
        warnings.append('foreground_fills_source_height_possible_neighboring_text')
    if image.width / max(1, x2 - x1) > 2 or image.height / ink_height > 3:
        warnings.append('excessive_source_whitespace')
    # Widely separated horizontal ink bands can indicate a second text line.
    projection = [mask.crop((0, y, image.width, y + 1)).getbbox() is not None for y in range(image.height)]
    runs, start = [], None
    for y, active in enumerate(projection + [False]):
        if active and start is None:
            start = y
        elif not active and start is not None:
            runs.append((start, y)); start = None
    substantial = [r for r in runs if r[1] - r[0] >= max(2, ink_height * 0.1)]
    if any(b[0] - a[1] > ink_height * 0.2 for a, b in zip(substantial, substantial[1:])):
        warnings.append('separated_ink_bands_possible_multiple_lines')
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    if size[0] + 2 * settings['padding'] > settings['max_width']:
        raise DatasetError("normalized line exceeds max_width")
    resized = image.resize(size, Image.Resampling.LANCZOS)
    ink = [round(x1 * size[0] / image.width), round(y1 * size[1] / image.height),
           round(x2 * size[0] / image.width), round(y2 * size[1] / image.height)]
    return resized, {'source_size': list(image.size), 'source_ink_bbox': list(bbox),
                     'source_text_height': ink_height, 'scale': scale,
                     'resized_size': list(size), 'resized_ink_bbox': ink, 'warnings': warnings,
                     'crop_diagnostic': diagnostic}

def build(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("mode") not in {"line", "paragraph", "page", "mixed"}:
        raise DatasetError("mode must be line, paragraph, page or mixed")
    if cfg["mode"] != "line":
        raise DatasetError("Legal paragraph/page integration requires verified full transcription and is not implemented")
    root = Path(cfg["source"]).expanduser().resolve()
    dest = Path(cfg["output"]).expanduser().resolve()
    if root == dest or root in dest.parents or dest in root.parents or dest.exists():
        raise DatasetError("Output must be a new directory outside the source")
    count = cfg.get("samples_per_split", 1)
    line_counts = cfg.get("lines_per_sample", [7, 8, 9, 10])
    counts = {s: count for s in ("train", "validation", "test")} if type(count) is int else count
    if not isinstance(counts, dict) or set(counts) != {"train", "validation", "test"} or any(type(v) is not int or v < 1 for v in counts.values()):
        raise DatasetError("samples_per_split must be positive or a positive count per split")
    sizes_by_split = line_counts if isinstance(line_counts, dict) else {
        split: [line_counts] if type(line_counts) is int else line_counts for split in counts
    }
    if set(sizes_by_split) != set(counts) or any(
        not isinstance(sizes, list) or not sizes or any(type(v) is not int or not 1 <= v <= 20 for v in sizes)
        for sizes in sizes_by_split.values()
    ):
        raise DatasetError("lines_per_sample must contain integers from 1 to 20 for each split")
    if cfg.get("content_type", "any") != "any":
        raise DatasetError("Content filtering requires reliable handwriting/printed source annotations; current reader has none")
    settings = {'target_text_height':48, 'min_source_text_height':8, 'max_scale':4, 'min_scale':0.7,
                'max_width':3000, 'max_height':2400, 'max_pixels':1048576, 'padding':16, 'line_spacing':12}
    alignment = cfg.get('alignment', 'left')
    if alignment not in ('left', 'center', 'right'):
        raise DatasetError('alignment must be left, center or right')
    supplied = cfg.get('quality', {})
    if not isinstance(supplied, dict) or set(supplied) - set(settings):
        raise DatasetError('Unknown quality settings')
    settings.update(supplied)
    if any(type(v) not in (int, float) or v <= 0 for v in settings.values()) or settings['max_scale'] < 1 or settings['min_scale'] > 1:
        raise DatasetError('Quality settings must be positive; max_scale must be at least 1')
    if any(type(settings[k]) is not int for k in ('target_text_height','min_source_text_height','max_width','max_height','max_pixels','padding','line_spacing')):
        raise DatasetError('Image dimensions and spacing must be integers')
    limit = cfg.get('candidate_limit', 500)
    if type(limit) is not int or limit < 1:
        raise DatasetError('candidate_limit must be positive')
    rejected = {s: {} for s in counts}
    decisions = {}
    if cfg.get('review_decisions'):
        review = json.loads(Path(cfg['review_decisions']).read_text(encoding='utf-8'))
        if not isinstance(review, list):
            raise DatasetError('review_decisions must be a JSON list')
        for decision in review:
            if not isinstance(decision, dict) or decision.get('recommendation') not in {'reject', 'review', 'accept_for_review'} or not isinstance(decision.get('source_path'), str) or not isinstance(decision.get('sha256'), str):
                raise DatasetError('Invalid reviewed crop decision')
            if decision['source_path'] in decisions:
                raise DatasetError('Duplicate reviewed crop decision')
            decisions[decision['source_path']] = decision
    groups, seen_ids, seen_pixels = {}, {}, {}
    for split in ("train", "validation", "test"):
        sizes = sizes_by_split[split]
        needed = sum(sizes[i % len(sizes)] for i in range(counts[split]))
        candidates = list(islice(iter_split(root, split), limit))
        if cfg.get('selection', 'sequential') == 'longest':
            def source_width(sample):
                with Image.open(sample.image) as source:
                    return source.width
            candidates.sort(key=source_width, reverse=True)
        elif cfg.get('selection', 'sequential') != 'sequential':
            raise DatasetError('selection must be sequential or longest')
        rows = []
        for sample in candidates:
            try:
                key = sample.image.relative_to(root).as_posix()
                decision = decisions.get(key)
                if decision:
                    digest = hashlib.sha256(sample.image.read_bytes()).hexdigest()
                    if digest != decision['sha256']:
                        raise DatasetError('reviewed source changed after visual review')
                    if decision['recommendation'] == 'reject':
                        rejected[split]['visually_confirmed_defect'] = rejected[split].get('visually_confirmed_defect', 0) + 1
                        continue
                with Image.open(sample.image) as image:
                    _, quality = normalize_line(image, settings)
                actionable = [w for w in quality['warnings'] if w != 'ink_near_source_edge_possible_clipping']
                if cfg.get('reject_flagged', False) and actionable:
                    raise DatasetError('flagged: ' + ', '.join(actionable))
            except (DatasetError, OSError) as exc:
                if isinstance(exc, DatasetError) and str(exc) == 'reviewed source changed after visual review':
                    raise
                reason = str(exc) if isinstance(exc, DatasetError) else 'unreadable image'
                rejected[split][reason] = rejected[split].get(reason, 0) + 1
                continue
            rows.append(sample)
            if len(rows) == needed:
                break
        if len(rows) != needed:
            raise DatasetError(f"{split}: insufficient acceptable source lines within candidate_limit; rejected={rejected[split]}")
        for sample in rows:
            identifier = sample.source_id or sample.image.relative_to(root).as_posix()
            with Image.open(sample.image) as image:
                pixels = ImageOps.exif_transpose(image).convert("RGB")
                pixels.load()
                digest = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
            for seen, key in ((seen_ids, identifier), (seen_pixels, digest)):
                if key in seen and seen[key] != split:
                    raise DatasetError(f"Source leakage between {seen[key]} and {split}")
                seen[key] = split
        groups[split] = rows
    dest.mkdir(parents=True)
    (dest / "images").mkdir()
    (dest / "previews").mkdir()
    emitted = {}
    comparisons = []
    flag_counts = {}
    rejected_composites = {}
    output_status_counts = {'info': 0, 'warning': 0, 'verified': 0}
    for split, rows in groups.items():
        sizes = sizes_by_split[split]
        records = []
        offset = 0
        for i in range(counts[split]):
            n = sizes[i % len(sizes)]
            group = rows[offset:offset+n]
            offset += n
            images, metrics = [], []
            for sample in group:
                with Image.open(sample.image) as image:
                    normalized, metric = normalize_line(image, settings)
                    images.append(normalized)
                    metrics.append(metric)
            width = max(im.width for im in images)
            pad, gap = settings['padding'], settings['line_spacing']
            canvas_width = width + 2 * pad
            canvas_height = sum(im.height for im in images) + gap * (n - 1) + 2 * pad
            if canvas_height > settings['max_height'] or canvas_width * canvas_height > settings['max_pixels']:
                rejected_composites[split] = rejected_composites.get(split, 0) + 1
                continue
            canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
            y, boxes = pad, []
            for im, metric in zip(images, metrics):
                offset_x = {'left': 0, 'center': (width-im.width)//2, 'right': width-im.width}[alignment]
                x = pad + offset_x
                canvas.paste(im, (x, y))
                boxes.append([x, y, x + im.width, y + im.height])
                x1,y1,x2,y2 = metric['resized_ink_bbox']
                metric['canvas_ink_bbox'] = [x1+x,y1+y,x2+x,y2+y]
                y += im.height + gap
            name = f"{split}-{i:05d}.png"
            canvas.save(dest / "images" / name)
            preview = canvas.copy()
            draw = ImageDraw.Draw(preview)
            for box in boxes:
                draw.rectangle(box, outline="red", width=2)
            preview.save(dest / "previews" / name)
            text = "\n".join(s.text for s in group)
            (dest / "previews" / f"{split}-{i:05d}.txt").write_text(text, encoding="utf-8")
            records.append({"image_path": f"images/{name}", "text": text, "split": split,
                            "sample_type": "line" if n == 1 else "synthetic_composite", "line_count": n,
                            "content_type": "unknown", "provenance": "source_split_only",
                            "source_ids": [s.source_id or s.image.relative_to(root).as_posix() for s in group],
                            "document_id": None, "page_id": None, "gt_verification": "source_labels_unmodified",
                            "line_boxes_xyxy": boxes, "line_quality": metrics,
                            "quality_status": "warning" if any(m['crop_diagnostic']['severity'] == 'warning' or m['warnings'] for m in metrics) else "info",
                            "before_dimensions": [max(m['source_size'][0] for m in metrics) + 24,
                                                  sum(m['source_size'][1] for m in metrics) + 12 * (n + 1)],
                            "after_dimensions": list(canvas.size)})
            output_status_counts[records[-1]['quality_status']] += 1
            for metric in metrics:
                for warning in metric['warnings']:
                    flag_counts[warning] = flag_counts.get(warning, 0) + 1
            comparisons.append({'sample': name, 'before_dimensions': records[-1]['before_dimensions'],
                                'after_dimensions': list(canvas.size),
                                'lines': [{'before_text_height': m['source_text_height'],
                                           'after_text_height': m['resized_ink_bbox'][3]-m['resized_ink_bbox'][1],
                                           'scale': m['scale'], 'source_size': m['source_size'],
                                           'after_size': m['resized_size'], 'warnings': m['warnings']} for m in metrics]})
        (dest / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        emitted[split] = len(records)
    report = {"counts": emitted, "source": str(root), "mode": "line",
              "lines_per_sample": sizes_by_split if isinstance(line_counts, dict) else sizes,
              "gt_coverage": "all composed source lines; source labels are not independently verified",
              "leakage_check": "selected source IDs and decoded pixels across source splits",
              "quality_settings": settings, "rejected_sources": rejected, "flagged_lines": flag_counts,
              "output_status_counts": output_status_counts,
              "rejected_composites_over_canvas_budget": rejected_composites,
              "quality_limitations": "Foreground threshold and band checks are heuristics; source clipping, neighboring text and GT correctness require manual review."}
    for name in ("manifest.json", "quality_report.json"):
        (dest / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / 'before_after.json').write_text(json.dumps({'samples':comparisons, 'rejected_sources':rejected,
                                                       'flagged_lines':flag_counts}, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
