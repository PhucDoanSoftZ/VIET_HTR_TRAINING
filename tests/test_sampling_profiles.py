import pytest

from types import SimpleNamespace

from htrtrain.sampling import WeightedTrainingMixin, sample_weights


def test_trainer_hook_samples_only_train_deterministically():
    pytest.importorskip('torch')

    class Base:
        def _get_eval_sampler(self, dataset):
            return dataset

    class Trainer(WeightedTrainingMixin, Base):
        ocr_training_weights = [0.0, .25, .75]
        args = SimpleNamespace(world_size=1, seed=42)
        train_dataset = range(3)

    trainer = Trainer()
    first = list(trainer._get_train_sampler())
    assert first == list(trainer._get_train_sampler()) and 0 not in first
    assert trainer._get_eval_sampler(range(4)) == range(4)
    with pytest.raises(ValueError, match='length'):
        trainer._get_train_sampler(range(4))


def test_explicit_weights_do_not_depend_on_group_size():
    profiles = {'v': {'single_line': .25, 'composite_7_10': .45,
                      'composite_11_13': .30, 'legal_full_page': 0}}
    assert sample_weights(['single_line', 'single_line', 'composite_7_10', 'composite_11_13'], profiles, 'v') == [.125, .125, .45, .30]
    with pytest.raises(ValueError, match='missing'):
        sample_weights(['single_line'], profiles, 'v')
    with pytest.raises(ValueError, match='explicitly'):
        sample_weights(['single_line'], profiles, 'unknown')
