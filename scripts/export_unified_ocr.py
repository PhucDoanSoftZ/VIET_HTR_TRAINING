"""Package a unified OCR dataset from local, audited inputs."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from PIL import Image
import yaml

from htrtrain.dataset import check_data, resolve_image
from htrtrain.legal_ocr import verified_document_text
from htrtrain.multiline import normalize_line
from htrtrain.sampling import sample_weights

SPLITS = ('train', 'validation', 'test')
RANGES = {'composite_7_10': (7, 10), 'composite_11_13': (11, 13)}
LINE = dict(target_text_height=36, min_source_text_height=8, max_scale=4,
            min_scale=.7, max_width=1250, padding=16)


def legal_pages(root: Path, curated: Path | None = None):
    for manifest in sorted(root.rglob('document.json')):
        doc = json.loads(manifest.read_text(encoding='utf-8'))
        pages = doc['pages']
        if not pages or [p['page_number'] for p in pages] != list(range(1, len(pages) + 1)):
            continue
        confirmed = False
        if curated is not None:
            try:
                verified_document_text(manifest.parent, curated)
                confirmed = True
            except ValueError:
                continue
        accepted = []
        for page in pages:
            folder = (manifest.parent / page['folder']).resolve()
            if not folder.is_relative_to(manifest.parent.resolve()):
                raise ValueError('Unsafe legal page path')
            meta = json.loads((folder / 'page.json').read_text(encoding='utf-8'))
            reminder = 'Text reading order and clipping need visual review; DOM geometry does not prove full GT coverage'
            if (not confirmed or meta.get('unresolved_text')
                    or any(w != reminder for w in meta.get('warnings', []))
                    or any(d['severity'] in ('error', 'warning') and d.get('detail') != reminder
                           for d in meta.get('diagnostics', []))
                    or meta.get('page_id') != page['page_id']
                    or meta.get('source_id') != doc['source_id']
                    or meta.get('document_id') != doc['document_id']):
                break
            image = folder / 'page.png'
            transcript = (folder / 'page.txt').read_text(encoding='utf-8')
            if (not transcript.strip() or transcript != '\n'.join(line['text'] for line in meta['lines'])
                    or hashlib.sha256(image.read_bytes()).hexdigest() != meta.get('render_sha256')):
                break
            accepted.append((image, {**meta, 'text': transcript,
                'field_gt': json.loads((folder / 'field_gt.json').read_text(encoding='utf-8')),
                'split': {'dev': 'validation', 'test': 'test', 'train': 'train'}[meta['split']],
                'gt_verification': 'human_confirmed_image_gt_match'}))
        if len(accepted) == len(pages):
            yield from accepted


def export(config: dict, *, dry_run: bool = False, smoke: bool = False) -> dict:
    source, output = Path(config['source']).resolve(), Path(config['output']).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output must be new and separate from source')
    quotas = config['quotas']
    if set(quotas) != set(SPLITS) or any(
        not isinstance(q, dict) or set(q) != set(RANGES)
        or any(type(n) is not int or n < 0 for n in q.values()) for q in quotas.values()
    ):
        raise ValueError('Provide nonnegative composite quotas by split and group')
    if type(config['seed']) is not int or type(config['max_canvas_pixels']) is not int or config['max_canvas_pixels'] <= 0:
        raise ValueError('Require an integer seed and positive integer canvas pixel budget')
    originals = {}
    for split in SPLITS:
        with (source / f'{split}.jsonl').open(encoding='utf-8') as stream:
            originals[split] = [row for line in stream if
                (row := json.loads(line)).get('sample_type') == 'original_line'
                and row.get('gt_verification') == 'provider_trusted_source'
                and row.get('quality_status') not in ('reject', 'rejected')]
        if not originals[split] or any(row['split'] != split for row in originals[split]):
            raise ValueError(f'Missing or inconsistent original split: {split}')
    legal_root = Path(config['legal_pages']).resolve() if config.get('legal_pages') else None
    if legal_root is not None and (not legal_root.is_dir() or output.is_relative_to(legal_root)):
        raise ValueError('Legal input must exist and be separate from output')
    curated = Path(config['legal_curated']) if config.get('legal_curated') else None
    legal = list(legal_pages(legal_root, curated)) if legal_root else []
    profiles, profile = config['profiles'], config['profile']
    sample_weights(list(RANGES) + ['single_line', 'legal_full_page'], profiles, profile)
    grouping = {}
    for split, records in originals.items():
        for row in records:
            for key in ('document_id', 'writer_id'):
                if row.get(key) is not None and grouping.setdefault((key, row[key]), split) != split:
                    raise ValueError(f'{key} occurs across source splits: {row[key]}')
    planned = {'originals': {s: len(v) for s, v in originals.items()}, 'quotas_per_split': quotas,
               'eligible_legal_pages': len(legal), 'profile': profile}
    if dry_run:
        planned['eligible_source_crops'] = {s: sum(bool(r.get('source_id')) for r in v) for s, v in originals.items()}
        planned['feasibility'] = {s: {g: ('insufficient distinct crops' if planned['eligible_source_crops'][s] < lo
                                        else 'candidate pool sufficient; geometry checked during build')
                                     for g, (lo, _) in RANGES.items()} for s in SPLITS}
        planned['legal_blocker'] = None if any(row['split'] == 'train' for _, row in legal) else 'No eligible reviewed legal training pages'
        planned['missing_positive_groups'] = [
            group for group, weight in profiles[profile].items() if weight and
            ((group in RANGES and not quotas['train'][group]) or
             (group == 'legal_full_page' and not any(row['split'] == 'train' for _, row in legal)))]
        return planned
    rng = random.Random(config['seed'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / 'dataset'
        rows = {s: [] for s in SPLITS}
        rejections = Counter()

        def emit(split, group, row, image):
            rel = Path('images') / group / f'{split}-{len(rows[split]):07d}.png'
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(image, Path):
                try:
                    os.link(image, dest)
                except OSError:
                    shutil.copyfile(image, dest)
            else:
                image.save(dest)
            checksum = hashlib.sha256(dest.read_bytes()).hexdigest()
            if row.get('local_file_sha256', checksum) != checksum:
                raise ValueError(f'Image checksum mismatch: {image}')
            with Image.open(dest) as loaded:
                size = loaded.size
            rows[split].append({**row, 'image_relpath': rel.as_posix(), 'image_path': rel.as_posix(),
                'sample_group': group, 'local_file_sha256': checksum, 'width': size[0], 'height': size[1]})

        for split, records in originals.items():
            if smoke:
                records = records[:40]
            pool = list({row['source_id']: row for row in records if row.get('source_id')}.values())
            for row in records:
                emit(split, 'single_line', row, resolve_image(source, row['image_path']))
            for group, (lo, hi) in RANGES.items():
                made = 0
                quota = min(quotas[split][group], 1) if smoke else quotas[split][group]
                seen, unusable = set(), set()
                for _ in range(quota * 12):
                    if made == quota or len(pool) < lo:
                        break
                    n = rng.randint(lo, min(hi, len(pool)))
                    batch = rng.sample(pool, n)
                    key = tuple(row['source_id'] for row in batch)
                    if key in seen or any(k in unusable for k in key):
                        continue
                    seen.add(key)
                    images, warnings = [], []
                    for row in batch:
                        try:
                            with Image.open(resolve_image(source, row['image_path'])) as crop:
                                image, metric = normalize_line(crop, LINE)
                            images.append(image)
                            warnings.append(sorted(set(metric['warnings'] + metric['crop_diagnostic']['reasons'])))
                        except (OSError, ValueError) as exc:
                            unusable.add(row['source_id'])
                            pool = [candidate for candidate in pool if candidate['source_id'] not in unusable]
                            rejections[str(exc)] += 1
                            break
                    if len(images) != n:
                        continue
                    width, height = max(i.width for i in images)+32, sum(i.height for i in images)+32+9*(n-1)
                    if width * height > config['max_canvas_pixels']:
                        rejections['canvas_pixel_limit'] += 1
                        continue
                    canvas, boxes, y = Image.new('RGB', (width, height), 'white'), [], 16
                    for image in images:
                        canvas.paste(image, (16, y))
                        boxes.append([16, y, 16+image.width, y+image.height])
                        y += image.height + 9
                    emit(split, group, {'split': split, 'text': '\n'.join(row['text'] for row in batch),
                        'source_ids': [row['source_id'] for row in batch], 'line_boxes_xyxy': boxes,
                        'line_count': n, 'line_quality_warnings': warnings,
                        'gt_verification': 'derived_from_provider_trusted_source'}, canvas)
                    made += 1
                if made < quota:
                    rejections[f'{split}:{group}:shortfall'] += quota-made
        for image, row in (legal[:1] if smoke else legal):
            emit(row['split'], 'legal_full_page', row, image)
        try:
            sample_weights([r['sample_group'] for r in rows['train']], profiles, profile)
        except ValueError as exc:
            raise ValueError(f'{exc}. Generation rejections and shortfalls: {dict(rejections)}') from exc
        for split, records in rows.items():
            (stage / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
        validation = check_data(stage)
        if not validation['ok']:
            raise ValueError(f'Unified dataset failed validation: {validation["splits"]}')
        planned['counts'] = {s: dict(Counter(r['sample_group'] for r in v)) for s, v in rows.items()}
        planned['shortfalls'] = {k: v for k, v in rejections.items() if k.endswith(':shortfall')}
        planned['split_limitation'] = 'Document/writer isolation cannot be inferred where identifiers are unavailable.'
        for name, payload in [('sampling_config', {'profiles': profiles, 'profile': profile}),
                              ('dataset_manifest', planned), ('quality_report', {'rejections': dict(rejections), 'validation': validation})]:
            (stage / f'{name}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        stage.rename(output)
    return planned


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--build', action='store_true')
    args = parser.parse_args()
    print(json.dumps(export(yaml.safe_load(args.config.read_text(encoding='utf-8')),
                            dry_run=args.dry_run, smoke=args.smoke), indent=2))
