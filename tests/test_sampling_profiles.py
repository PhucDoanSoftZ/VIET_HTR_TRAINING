import pytest

from htrtrain.sampling import sample_weights


def test_explicit_weights_do_not_depend_on_group_size():
    profiles = {'v': {'single_line': .25, 'composite_7_10': .45,
                      'composite_11_13': .30, 'legal_full_page': 0}}
    assert sample_weights(['single_line', 'single_line', 'composite_7_10', 'composite_11_13'], profiles, 'v') == [.125, .125, .45, .30]
    with pytest.raises(ValueError, match='missing'):
        sample_weights(['single_line'], profiles, 'v')
    with pytest.raises(ValueError, match='explicitly'):
        sample_weights(['single_line'], profiles, 'unknown')
