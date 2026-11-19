"""Conservative crop diagnostics; a warning is not a human GT review."""
from collections import deque
from statistics import median
from PIL import Image, ImageOps

def analyze(image):
    gray = ImageOps.grayscale(ImageOps.exif_transpose(image))
    width, height = gray.size
    if not width or not height:
        return {'severity':'reject', 'reasons':['empty_image']}, gray
    pixels = gray.load()
    border = [pixels[x, y] for y in {0, height - 1} for x in range(width)]
    border += [pixels[x, y] for x in {0, width - 1} for y in range(height)]
    bg = median(border)
    # Estimate polarity from the border; nonuniform or photographic backgrounds remain ambiguous.
    polarity = 'light_on_dark' if bg < 110 else 'dark_on_light'
    threshold = min(255, int(bg) + 35) if polarity == 'light_on_dark' else max(0, int(bg) - 35)
    mask = gray.point(lambda value: 255 if (value > threshold if polarity == 'light_on_dark' else value < threshold) else 0)
    small = mask.copy()
    small.thumbnail((480, 120), Image.Resampling.NEAREST)
    w, h = small.size
    data = bytearray(small.tobytes())
    components = []
    for index in range(w * h):
        if not data[index]:
            continue
        data[index] = 0
        queue = deque([index]); n = 0; xs = []; ys = []
        while queue:
            point = queue.popleft(); x, y = point % w, point // w
            n += 1; xs.append(x); ys.append(y)
            for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                k = ny*w+nx
                if 0 <= nx < w and 0 <= ny < h and data[k]:
                    data[k] = 0; queue.append(k)
        if n >= 3:
            components.append([min(xs), min(ys), max(xs)+1, max(ys)+1, n])
    bbox = mask.getbbox()
    reasons, severity = [], 'info'
    if width < 8 or height < 8:
        severity, reasons = 'reject', ['image_dimensions_too_small']
    elif not bbox or not components:
        severity, reasons = 'warning', ['no_reliable_foreground_detected']
    else:
        x1,y1,x2,y2 = bbox
        if x1 <= 1 or y1 <= 1 or x2 >= width-1 or y2 >= height-1:
            reasons.append('foreground_near_edge_review')
        eligible = [c for c in components if (c[2]-c[0]) < w * .65 and (c[3]-c[1]) >= 2]
        if len(eligible) < 2:
            reasons.append('insufficient_separate_components')
        if (y2-y1) / height > .9:
            reasons.append('foreground_spans_crop_height_review')
        if len(eligible) >= 2:
            centers = sorted((c[1]+c[3])/2 for c in eligible)
            if any(b-a > h*.28 for a,b in zip(centers, centers[1:])):
                reasons.append('separated_component_bands_review')
        if reasons:
            severity = 'warning'
    return {'severity':severity, 'reasons':reasons, 'border_median':bg,
            'border_threshold':threshold, 'polarity':polarity, 'foreground_bbox':list(bbox) if bbox else None,
            'component_count':len(components), 'component_sample_size':[w,h],
            'method':'border median threshold and bounded connected components; no GT verification'}, mask
