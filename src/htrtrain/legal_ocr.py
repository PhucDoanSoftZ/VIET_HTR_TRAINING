"""Render a bounded legal OCR example directly from an archived HTML document.

All text is DOM-derived. Output is provisional pending visual review.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse
from PIL import Image, ImageChops, ImageDraw, ImageStat

def map_assets(html: str, asset_dir: Path) -> tuple[str, list[str]]:
    """Replace archived local URLs only with same-named files from an allowed directory."""
    warnings = []
    root = asset_dir.resolve()
    def replace(match):
        url = match.group(0)
        name = PurePosixPath(unquote(urlparse(url).path)).name
        target = (root / name).resolve()
        if target.parent == root and target.is_file():
            return target.as_uri()
        warnings.append(f"unresolved local asset: {name}")
        return "about:blank#missing-asset"
    return re.sub(r"file://[^\s\"'\)]+", replace, html, flags=re.I), warnings

def rows_to_lines(rows: list[dict], width: int, height: int) -> tuple[list[dict], list[str]]:
    warnings = []
    lines = []
    for row in sorted(rows, key=lambda item: (item["y"], item["x"])):
        x, y, w, h = (row[k] for k in ("x", "y", "w", "h"))
        if not all(math.isfinite(v) for v in (x, y, w, h)) or not (w > 0 and h > 0 and 0 <= x < x+w <= width and 0 <= y < y+h <= height):
            warnings.append("text outside page or invalid bounds")
            continue
        if row.get("ambiguous"):
            warnings.append(row["ambiguous"])
        region = row.get("column")
        table_row = row.get("table_row")
        candidates = [
            line for line in lines if line["column"] == region
            and line["table_row"] == table_row
            and min(line["anchor_y"] + line["h"], y+h) - max(line["anchor_y"], y) >= 0.35 * min(line["h"], h)
        ]
        line = min(candidates, key=lambda line: abs(line["anchor_y"] + line["h"] - y - h), default=None)
        if line is None:
            line = {"y": y, "h": h, "anchor_y": y, "column": region, "table_row": table_row, "words": []}
            lines.append(line)
        line["words"].append(row)
    lines.sort(key=lambda line: (
        line["words"][0].get("column_top", line["y"]),
        line["column"] if line["column"] is not None else -1,
        line["y"],
    ))
    result = []
    for line in lines:
        words = sorted(line["words"], key=lambda row: row["x"])
        x1 = min(r["x"] for r in words); y1 = min(r["y"] for r in words)
        x2 = max(r["x"] + r["w"] for r in words); y2 = max(r["y"] + r["h"] for r in words)
        parts = []
        for i, word in enumerate(words):
            previous = words[i - 1] if i else None
            # Preserve adjacent markup, e.g. <b>12</b><span>/05</span>.
            if previous and (word.get("separator", True) or word["x"] - previous["x"] - previous["w"] > 1):
                parts.append(" ")
            parts.append(word["text"])
        result.append({"text": "".join(parts), "bbox": [x1, y1, x2, y2], "words": words})
    return result, sorted(set(warnings))

DOM_WORDS = r"""pageIndex => {
 const p=document.querySelectorAll('section.page')[pageIndex ?? 0], origin=p.getBoundingClientRect();
 const walker=document.createTreeWalker(p,NodeFilter.SHOW_TEXT);
 const out=[], unresolved=[], excluded=[], columnByElement=new Map(), fieldInstances=new Map(), tableRows=new Map();
 const visible=el=>{
  if(getComputedStyle(el).visibility!=='visible' || el.closest('script,style,defs,[hidden]')) return false;
  for(let a=el;a;a=a.parentElement){
   const s=getComputedStyle(a);
   if(s.display==='none' || +s.opacity===0 || s.contentVisibility==='hidden') return false;
  }
  return true;
 };
 const box=r=>({x:r.left-origin.left,y:r.top-origin.top,w:r.width,h:r.height});
 const inPage=r=>r.left>=origin.left && r.top>=origin.top && r.right<=origin.right && r.bottom<=origin.bottom;
 const decoration='svg.stamp,svg.signature,[data-ocr-decorative]';
 for(const el of p.querySelectorAll(decoration)){
  if(!visible(el) || el.parentElement.closest(decoration)) continue;
  const r=el.getBoundingClientRect();
  if(r.width<=0 || r.height<=0 || r.right<=origin.left || r.left>=origin.right ||
     r.bottom<=origin.top || r.top>=origin.bottom) continue;
  excluded.push({kind:el.matches('.stamp')?'stamp':el.matches('.signature')?'signature':'decorative',
    text:[...el.querySelectorAll('text')].filter(visible).map(e=>e.textContent.trim()).filter(Boolean).join('\n'),
    text_origin:'SVG DOM inventory',...box(r),partially_outside_page:!inPage(r)});
 }
 // Split side-by-side block columns, but keep single-row fields and table cells in row order.
 const columnWarnings=[];
 let columnId=0;
 for(const el of p.querySelectorAll('*')){
  if(!visible(el) || el.closest('table,svg,[data-field]')) continue;
  const s=getComputedStyle(el);
  if(!['flex','grid'].includes(s.display) || s.flexDirection==='column' || el.children.length<2) continue;
  const children=[...el.children].filter(c=>visible(c) && c.innerText?.trim())
   .map(child=>({child,rect:child.getBoundingClientRect()})).sort((a,b)=>a.rect.left-b.rect.left);
  if(children.length<2) continue;
  const top=Math.min(...children.map(c=>c.rect.top));
  if(!children.every((c,i)=>c.rect.width>0 && Math.abs(c.rect.top-top)<8 &&
       (!i || children[i-1].rect.right<=c.rect.left+1))) continue;
  const multiline=children.some(({child,rect})=>{
   const style=getComputedStyle(child);
   return rect.height>1.65*(parseFloat(style.lineHeight)||1.2*parseFloat(style.fontSize));
  });
  if(!multiline || children.some(({child})=>child.closest('[data-field]'))) continue;
  if(children.some(({child})=>columnByElement.has(child) || [...columnByElement.keys()].some(c=>c.contains(child)))) {
   columnWarnings.push('Nested columns require visual review');
   continue;
  }
  children.forEach(({child})=>columnByElement.set(child,{column:columnId++,column_top:top-origin.top}));
 }
 let node, previousNode;
 while(node=walker.nextNode()) {
  const el=node.parentElement;
  if(!visible(el) || el.closest(decoration)) continue;
  const style=getComputedStyle(el);
  if(style.color==='rgba(0, 0, 0, 0)' || style.webkitTextFillColor==='rgba(0, 0, 0, 0)') continue;
  const field=el.closest('[data-field]');
  if(field && !fieldInstances.has(field)) fieldInstances.set(field, fieldInstances.size);
  const tableRow=el.closest('tr');
  if(tableRow && !tableRows.has(tableRow)) tableRows.set(tableRow, tableRows.size);
  let column={};
  for(let a=el;a && a!==p;a=a.parentElement) if(columnByElement.has(a)){column=columnByElement.get(a);break;}
  const text=node.textContent, re=/\S+/gu; let m;
  while ((m=re.exec(text))) {
   const r=document.createRange(); r.setStart(node,m.index); r.setEnd(node,m.index+m[0].length);
   const rects=[...r.getClientRects()].filter(x=>x.width>0 && x.height>0);
   if(rects.length) {
    const rect={left:Math.min(...rects.map(x=>x.left)),top:Math.min(...rects.map(x=>x.top)),
                right:Math.max(...rects.map(x=>x.right)),bottom:Math.max(...rects.map(x=>x.bottom))};
    rect.width=rect.right-rect.left; rect.height=rect.bottom-rect.top;
    let clipped=!inPage(rect); for(let a=el;a && a!==p;a=a.parentElement) {
     const s=getComputedStyle(a), b=a.getBoundingClientRect();
     if((/hidden|clip|scroll|auto/.test(s.overflowX) && (rect.left<b.left-1 || rect.right>b.right+1)) ||
        (/hidden|clip|scroll|auto/.test(s.overflowY) && (rect.top<b.top-1 || rect.bottom>b.bottom+1))) clipped=true;
    }
    const token={text:m[0],...box(rect),...column,field:field?.dataset.field,
       table_row:tableRow ? tableRows.get(tableRow) : null,
       font_family:style.fontFamily, font_size:style.fontSize,
       field_instance:fieldInstances.get(field),dom_index:out.length+unresolved.length,
       separator:m.index>0 || !previousNode || /\s$/.test(previousNode.textContent),
       ambiguous:rects.length>1?'word spans multiple visual rectangles':(el.closest('svg')?'Ordinary SVG text geometry requires visual review':
         (style.writingMode!=='horizontal-tb' || style.transform!=='none' ? 'Rotated or vertical text requires visual review' : null))};
    if(clipped) unresolved.push({...token,reason:'clipped_or_outside_page'});
    else if(rects.length>1 && !el.closest('svg') &&
            rects.some(r=>r.top>=rects[0].bottom)) unresolved.push({...token,reason:'token_wraps_across_lines'});
    else out.push(token);
   }
  }
  previousNode=node;
 }
 let pseudo=0; for(const el of p.querySelectorAll('*')) for(const part of ['::before','::after']) {
  const s=getComputedStyle(el,part); if(s.content && !['none','normal','""',"''"].includes(s.content) && s.display!=='none' && s.visibility!=='hidden') pseudo++;
 }
 return {width:origin.width,height:origin.height,rows:out,unresolved,excluded,columnWarnings,
         columns:columnByElement.size, svg:p.querySelectorAll('svg').length,
         images:p.querySelectorAll('img,canvas').length, pseudo,
         fonts:[...document.fonts].map(f=>({family:f.family,status:f.status,style:f.style,weight:f.weight}))};
}"""

def field_coverage(fields, lines, page_number=1):
    """Compare rendered field instances with source display text without substituting labels."""
    instances = {}
    for line in lines:
        for word in line["words"]:
            if word.get("field") is not None:
                instances.setdefault((word["field"], word["field_instance"]), []).append(word)
    visible = {}
    for (name, _), words in instances.items():
        fragments = []
        for word in sorted(words, key=lambda item: item["dom_index"]):
            if fragments and word["separator"]:
                fragments.append(" ")
            fragments.append(word["text"])
        visible.setdefault(name, []).append("".join(fragments))
    results = []
    for field in fields:
        expected = field["text"]
        pages = {box["page"] for box in field.get("bbox", [])}
        if not isinstance(expected, str) or not expected.strip() or page_number not in pages:
            continue
        values = visible.get(field["name"], [])
        if any(" ".join(value.split()) == " ".join(expected.split()) for value in values):
            status = "matched"
        elif not values:
            status = "not_included"
        else:
            status = "mismatch_or_partial"
        results.append({"name": field["name"], "expected_display_text": expected, "included_instances": values,
                        "source_pages": sorted(pages), "status": status})
    return {"scope": "nonempty source field display text with a bbox on this page; whitespace collapsed for comparison",
            "counts": {status: sum(row["status"] == status for row in results)
                       for status in ("matched", "not_included", "mismatch_or_partial")},
            "fields": results}

def generate(zip_path: Path, output: Path, asset_dir: Path, dossier: str | None = None, document: str | None = None,
             *, page_number=1, all_pages=False) -> dict:
    from playwright.sync_api import sync_playwright
    if output.exists():
        raise ValueError("Output must be a new directory")
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith('/ground_truth.json'))
        if dossier:
            names = [n for n in names if PurePosixPath(n).parent.name == dossier]
        if not names:
            raise ValueError("No matching dossier in ZIP")
        gt_name = names[0]
        gt = json.loads(archive.read(gt_name))
        docs = [d for d in gt['documents'] if d.get('files', {}).get('html') and d['files'].get('pages')]
        if document:
            docs = [d for d in docs if d['doc_id'] == document]
        if not docs:
            raise ValueError("No matching rendered document in dossier")
        doc = docs[0]; base = PurePosixPath(gt_name).parent
        def entry(name):
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError("Unsafe ZIP document path")
            return str(base / path)
        html = archive.read(entry(doc['files']['html'])).decode('utf-8')
        page_count = len(doc['files']['pages'])
        if not 1 <= page_number <= page_count or (all_pages and page_number != 1):
            raise ValueError('Page request is outside the archived document')
        references = [archive.read(entry(name)) for name in (
            doc['files']['pages'] if all_pages else [doc['files']['pages'][page_number - 1]])]
    html, warnings = map_assets(html, asset_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        file = Path(temp) / 'document.html'
        file.write_text(html, encoding='utf-8')
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(device_scale_factor=200/96, java_script_enabled=False)
            page.goto(file.as_uri(), wait_until='load')
            page.evaluate('document.fonts.ready')
            sections = page.locator('section.page')
            if sections.count() != page_count:
                raise ValueError('HTML page count differs from archive')
            captures = []
            for number in (range(1, page_count + 1) if all_pages else [page_number]):
                index = number - 1
                section = sections.nth(index)
                section.scroll_into_view_if_needed()
                dom = page.evaluate(DOM_WORDS, index)
                png = section.screenshot()
                settled = page.evaluate(DOM_WORDS, index)
                if settled != dom:
                    dom = settled
                    if section.screenshot() != png or page.evaluate(DOM_WORDS, index) != dom:
                        raise ValueError('DOM text or geometry changed during screenshot')
                captures.append((number, dom, png))
            pdf = page.pdf(prefer_css_page_size=True, print_background=True) if all_pages else None
            browser.close()
    if all_pages:
        import pymupdf
        with pymupdf.open(stream=pdf, filetype='pdf') as rendered_pdf:
            if len(rendered_pdf) != page_count:
                raise ValueError(f'PDF has {len(rendered_pdf)} pages, HTML has {page_count}; no inferred page mapping')
            output.mkdir()
            (output / 'document.pdf').write_bytes(pdf)
            pages = []
            for (number, dom, png), reference in zip(captures, references, strict=True):
                folder = output / f'p{number:03d}'
                meta = write_page(folder, gt, doc, number, dom, png, reference, warnings.copy(), asset_dir)
                pdf_page = rendered_pdf[number - 1]
                raster = pdf_page.get_pixmap(dpi=200, alpha=False)
                raster.save(folder / 'pdf_page.png')
                audit = compare_pdf_text(pdf_page.get_text(), (folder / 'page.txt').read_text(encoding='utf-8'))
                audit['pdf_raster_size'] = [raster.width, raster.height]
                audit['pdf_raster_status'] = 'comparison_only'
                (folder / 'pdf_comparison.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
                pages.append({'page_number': number, 'page_id': meta['page_id'], 'folder': folder.name,
                              'pdf_text_audit': audit, 'gt_status': meta['gt_status']})
        report = {'document_id': doc['doc_id'], 'source_id': gt['dossier_id'], 'pages': pages,
                  'pdf_sha256': hashlib.sha256(pdf).hexdigest(),
                  'document_gt_status': 'provisional_page_review_required'}
        (output / 'document.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (output / 'document.provisional.txt').write_text(
            '\n\f\n'.join((output / row['folder'] / 'page.txt').read_text(encoding='utf-8') for row in pages),
            encoding='utf-8')
        return report
    number, dom, png = captures[0]
    meta = write_page(output, gt, doc, number, dom, png, references[0], warnings, asset_dir)
    return {'output':str(output), 'lines':len(meta['lines']), 'warnings':meta['warnings'], 'comparison':meta['comparison']}


def compare_pdf_text(pdf_text, dom_text):
    """Compare text layer inventories without using PDF extraction as a label."""
    missing = Counter(dom_text.split()) - Counter(pdf_text.split())
    extra = Counter(pdf_text.split()) - Counter(dom_text.split())
    return {'missing_main_body_tokens': sum(missing.values()), 'extra_pdf_tokens': sum(extra.values()),
            'missing_examples': list(missing)[:20], 'extra_examples': list(extra)[:20],
            'whitespace_normalized_order_match': pdf_text.split() == dom_text.split()}


def verified_document_text(document_dir, curated_dir):
    """Assemble a transcript only when existing human reviews cover every exact page."""
    doc = json.loads((document_dir / 'document.json').read_text(encoding='utf-8'))
    pages = doc['pages']
    if not pages or [row['page_number'] for row in pages] != list(range(1, len(pages) + 1)):
        raise ValueError('Document manifest must list consecutive pages')
    accepted = {}
    for split in ('train', 'validation', 'test'):
        for line in (curated_dir / f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('sample_type') != 'page' or row.get('gt_verification') != 'human_confirmed_image_gt_match':
                continue
            key = row['page_id']
            if key in accepted:
                raise ValueError(f'Duplicate confirmed page: {key}')
            accepted[key] = (split, row)
    pieces, splits = [], set()
    for entry in pages:
        page_dir = document_dir / entry['folder']
        page = json.loads((page_dir / 'page.json').read_text(encoding='utf-8'))
        text = (page_dir / 'page.txt').read_text(encoding='utf-8')
        split, row = accepted.get(entry['page_id'], (None, {}))
        expected_split = {'train': 'train', 'dev': 'validation', 'test': 'test'}.get(page['split'])
        image_hash = hashlib.sha256((page_dir / 'page.png').read_bytes()).hexdigest()
        if (row.get('source_sha256'), row.get('text'), row.get('source_id'), row.get('document_id'), split) != (
            image_hash, text, doc['source_id'], doc['document_id'], expected_split
        ):
            raise ValueError(f"Missing or mismatched human confirmation for {entry['page_id']}")
        pieces.append(text)
        splits.add(split)
    if len(splits) != 1:
        raise ValueError('Document pages cross dataset splits')
    return '\n\f\n'.join(pieces)


def write_page(output, gt, doc, page_number, dom, png, reference, warnings, asset_dir):
    image = Image.open(io.BytesIO(png)).convert('RGB')
    scale_x, scale_y = image.width/dom['width'], image.height/dom['height']
    def image_coordinates(row):
        return {**row, 'x':row['x']*scale_x, 'y':row['y']*scale_y,
                'w':row['w']*scale_x, 'h':row['h']*scale_y,
                **({'column_top': row['column_top']*scale_y} if 'column_top' in row else {})}

    rows = [image_coordinates(row) for row in dom['rows']]
    lines, flags = rows_to_lines(rows, image.width, image.height)
    warnings.extend(flags)
    warnings.extend(dom['columnWarnings'])
    included = {word['dom_index'] for line in lines for word in line['words']}
    unresolved = [image_coordinates(row) for row in dom['unresolved']]
    unresolved.extend({**row, 'reason':'invalid_image_bbox'} for row in rows if row['dom_index'] not in included)
    if unresolved:
        warnings.append('Ordinary text with clipped or unsupported geometry retained in unresolved_text')
    if dom['images']: warnings.append('Embedded image or canvas may contain untranscribed text')
    if dom['pseudo']: warnings.append('CSS generated content may contain untranscribed text')
    warnings.append('Text reading order and clipping need visual review; DOM geometry does not prove full GT coverage')
    if not lines:
        raise ValueError('No visible text rows extracted')
    ref = Image.open(io.BytesIO(reference)).convert('RGB')
    same_size = ref.size == image.size
    difference = None
    if same_size:
        difference = sum(ImageStat.Stat(ImageChops.difference(image, ref)).mean) / 3
    else:
        warnings.append('Rendered page dimensions differ from ZIP reference')
    if difference is not None and difference > 0.5:
        warnings.append('Rendered page pixels differ from ZIP reference')
    text = '\n'.join(line['text'] for line in lines)
    page_id = f"{gt['dossier_id']}--{doc['doc_id']}--p{page_number}"
    bucket = int(hashlib.sha256(f"docbench-v1:{gt['dossier_id']}".encode()).hexdigest()[:8], 16) % 100
    meta = {'source_id':gt['dossier_id'], 'document_id':doc['doc_id'], 'document_type':doc['doc_type'],
            'page_id':page_id, 'page_number':page_number,
            'split':'test' if bucket < 20 else 'dev', 'split_policy':'benchmark docbench-v1 case hash',
            'gt_status':'provisional_manual_review_required',
            'warnings':sorted(set(warnings)), 'asset_directory':str(asset_dir), 'lines':lines,
            'comparison':{'reference_size':list(ref.size), 'rendered_size':list(image.size), 'mean_absolute_pixel_difference':difference,
                          'reference_sha256':hashlib.sha256(reference).hexdigest()},
            'gt_origin':'visible DOM text from same HTML state as screenshot; not field GT or model OCR'}
    meta['render_sha256'] = hashlib.sha256(png).hexdigest()
    meta['excluded_regions'] = [image_coordinates(row) for row in dom['excluded']]
    meta['unresolved_text'] = unresolved
    meta['field_coverage'] = field_coverage(doc.get('fields', []), lines, page_number)
    fills = {field['name']: field.get('fill') for field in doc.get('fields', [])}
    for line in lines:
        for word in line['words']:
            word['text_role'] = 'filled_field' if word.get('field') else 'static_text'
            word['field_fill'] = fills.get(word.get('field'))
    meta['full_page_ocr_coverage'] = {
        'included_dom_tokens': len(included), 'unresolved_dom_tokens': len(unresolved),
        'column_regions': dom['columns'], 'excluded_regions': len(dom['excluded']),
        'css_generated_regions': dom['pseudo'], 'embedded_image_or_canvas_regions': dom['images'],
        'visual_verification': 'pending', 'scope': 'visible main-body DOM text; stamps and signatures excluded',
    }
    meta['variant'] = doc.get('variant')
    meta['font_faces'] = dom.get('fonts', [])
    handwriting = sorted({word.get('font_family', '') for line in lines for word in line['words']
                          if word['field_fill'] == 'handwritten'})
    diagnostics = []
    for field in meta['field_coverage']['fields']:
        if field['status'] != 'matched':
            diagnostics.append({'severity': 'warning', 'code': 'field_not_matched', 'field': field['name'],
                                'detail': field['status']})
    if unresolved:
        diagnostics.append({'severity': 'error', 'code': 'unresolved_text', 'count': len(unresolved)})
    for warning in sorted(set(warnings)):
        diagnostics.append({'severity': 'warning', 'code': 'render_warning', 'detail': warning})
    if handwriting:
        diagnostics.append({'severity': 'warning', 'code': 'handwriting_font_unverified', 'families': handwriting,
                            'detail': 'Computed CSS family does not prove the actual font used; visual review required.'})
    diagnostics.append({'severity': 'info', 'code': 'excluded_decorations', 'count': len(meta['excluded_regions'])})
    meta['diagnostics'] = diagnostics
    output.mkdir()
    (output / 'page.png').write_bytes(png)
    # Reference is for visual comparison only; GT and boxes describe page.png.
    (output / 'reference.png').write_bytes(reference)
    (output / 'page.txt').write_text(text, encoding='utf-8')
    (output / 'field_gt.json').write_text(json.dumps({'source_id': gt['dossier_id'], 'document_id': doc['doc_id'],
        'page_number': page_number, 'fields': doc.get('fields', []), 'coverage': meta['field_coverage']},
        ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'diagnostics.json').write_text(json.dumps({'status': meta['gt_status'], 'diagnostics': diagnostics},
        ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'page.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    # Include all lines intersecting the crop; never infer a paragraph from field labels.
    chosen = lines[4:6] if len(lines) >= 8 else lines[:min(2, len(lines))]
    # Prefer a compact crop showing both printed labels and a filled handwriting field.
    for start in range(len(lines) - 1):
        candidate = lines[start:start + 2]
        words = [word for line in candidate for word in line['words']]
        if (any(word['field_fill'] == 'handwritten' for word in words)
            and any(word['text_role'] == 'static_text' or word['field_fill'] == 'printed' for word in words)
            and max(line['bbox'][3] for line in candidate) - min(line['bbox'][1] for line in candidate) <= image.height / 4):
            chosen = candidate
            break
    y1 = max(0, int(min(l['bbox'][1] for l in chosen))-2)
    y2 = min(image.height, int(max(l['bbox'][3] for l in chosen))+3)
    # Keep whole intersecting text boxes, including those in adjacent columns.
    while True:
        selected = [l for l in lines if l['bbox'][1] < y2 and l['bbox'][3] > y1]
        bounds = (math.floor(min(y1, *(l['bbox'][1] for l in selected))),
                  math.ceil(max(y2, *(l['bbox'][3] for l in selected))))
        if bounds == (y1, y2):
            break
        y1, y2 = bounds
    crop = image.crop((0, y1, image.width, y2))
    crop.save(output / 'paragraph.png')
    (output / 'paragraph.txt').write_text('\n'.join(l['text'] for l in selected), encoding='utf-8')
    paragraph = {**meta, 'crop_page_xyxy':[0,y1,image.width,y2], 'lines':selected}
    paragraph['field_coverage'] = field_coverage(doc.get('fields', []), selected, page_number)
    paragraph['full_page_ocr_coverage'] = None
    paragraph['crop_ocr_coverage'] = {'included_dom_tokens': sum(len(line['words']) for line in selected),
                                     'visual_verification': 'pending'}
    (output / 'paragraph.json').write_text(json.dumps(paragraph, ensure_ascii=False, indent=2), encoding='utf-8')
    preview = image.copy(); draw = ImageDraw.Draw(preview)
    for line in lines: draw.rectangle(line['bbox'], outline='red', width=2)
    preview.save(output / 'page_preview.png')
    preview.crop((0, y1, image.width, y2)).save(output / 'paragraph_preview.png')
    return meta

def main():
    import sys

    if sys.argv[1:2] == ['assemble-verified']:
        assemble = argparse.ArgumentParser()
        assemble.add_argument('--document-dir', type=Path, required=True)
        assemble.add_argument('--curated-dir', type=Path, required=True)
        assemble.add_argument('--output', type=Path, required=True)
        args = assemble.parse_args(sys.argv[2:])
        text = verified_document_text(args.document_dir, args.curated_dir)
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        return
    p = argparse.ArgumentParser()
    p.add_argument('--zip', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--asset-dir', type=Path, required=True); p.add_argument('--dossier'); p.add_argument('--document')
    pages = p.add_mutually_exclusive_group()
    pages.add_argument('--page', type=int, default=1)
    pages.add_argument('--all-pages', action='store_true', help='Export every page and compare the Chromium PDF with DOM GT (requires PyMuPDF)')
    args = p.parse_args()
    print(json.dumps(generate(args.zip, args.output, args.asset_dir, args.dossier, args.document,
                              page_number=args.page, all_pages=args.all_pages), ensure_ascii=True, indent=2))

if __name__ == '__main__':
    main()
