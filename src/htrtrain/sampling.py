"""Validate explicit sample-group weights for future train-only sampling."""
import math
from collections import Counter
import json

from .dataset import jsonl_path

GROUPS = frozenset({'single_line', 'composite_7_10', 'composite_11_13', 'legal_full_page'})


def training_weights(root):
    path = root / 'sampling_config.json'
    if not path.is_file():
        return None
    config = json.loads(path.read_text(encoding='utf-8'))
    with jsonl_path(root, 'train').open(encoding='utf-8') as stream:
        groups = [json.loads(line)['sample_group'] for line in stream]
    return sample_weights(groups, config['profiles'], config['profile'])


def sample_weights(groups: list[str], profiles: dict, profile: str) -> list[float]:
    """Assign per-record weights; fail rather than redistribute missing groups."""
    if profile not in profiles:
        raise ValueError('An existing sampling profile must be selected explicitly')
    weights = profiles[profile]
    if not isinstance(weights, dict) or set(weights) != GROUPS:
        raise ValueError('Profile must specify every sample group')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('Sampling weights must be finite and nonnegative')
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError('Sampling weights must sum to one')
    if any(group not in GROUPS for group in groups):
        raise ValueError('Unknown sample group')
    counts = Counter(groups)
    missing = [group for group, value in weights.items() if value > 0 and not counts[group]]
    if missing:
        raise ValueError(f'Positive-weight groups missing from training data: {sorted(missing)}')
    return [weights[group] / counts[group] for group in groups]
