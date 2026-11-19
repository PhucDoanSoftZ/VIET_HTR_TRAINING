import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageDraw
import yaml

from htrtrain.sampling import training_weights

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('export_ocr', ROOT / 'scripts/export_unified_ocr.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_smoke_export_and_weights(tmp_path):
    source = tmp_path / 'source'
    (source / 'images').mkdir(parents=True)
    for split_index, split in enumerate(builder.SPLITS):
        rows = []
        for index in range(20):
            name = f'images/{split}-{index}.png'
            image = Image.new('RGB', (120, 60), 'white')
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 15, 40, 44), fill='black')
            draw.rectangle((65, 15, 85 + index, 44), fill=(split_index * 10, 0, 0))
            image.save(source / name)
            rows.append(dict(image_path=name, split=split, sample_type='original_line',
                             gt_verification='provider_trusted_source', quality_status='info',
                             source_id=f'{split}:{index}', text=f'Dòng {index}'))
        (source / f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    config = yaml.safe_load((ROOT / 'configs/unified_ocr_profiles.yaml').read_text())
    out = tmp_path / 'export'
    config.update(source=str(source), output=str(out))
    assert builder.export(config, dry_run=True)['originals']['train'] == 20
    assert not out.exists()
    report = builder.export(config, smoke=True)
    for split in builder.SPLITS:
        assert report['counts'][split] == {'single_line': 20, 'composite_7_10': 1, 'composite_11_13': 1}
        rows = [json.loads(line) for line in (out / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in rows[-2:]:
            assert row['text'].splitlines() == [f"Dòng {key.split(':')[1]}" for key in row['source_ids']]
            assert all(key.startswith(split + ':') for key in row['source_ids'])
            boxes = row['line_boxes_xyxy']
            assert all(a[3] < b[1] for a, b in zip(boxes, boxes[1:]))
            assert all(0 < x < x2 < row['width'] and 0 < y < y2 < row['height'] for x,y,x2,y2 in boxes)
    assert training_weights(out)[-2:] == [.45, .30]


def test_provisional_legal_excluded(tmp_path):
    page = tmp_path / 'p1'
    page.mkdir()
    (tmp_path / 'document.json').write_text(json.dumps({'pages': [{'page_number': 1, 'folder': 'p1'}]}))
    (page / 'page.json').write_text(json.dumps({'gt_status': 'provisional_manual_review_required'}))
    assert list(builder.legal_pages(tmp_path)) == []
