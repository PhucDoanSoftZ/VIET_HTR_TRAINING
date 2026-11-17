from htrtrain.evaluate import normalize, score


def test_cer_wer_and_unicode_normalization():
    assert normalize(" Tiếng   Việt ") == "Tiếng Việt"
    assert score(["abc def"], ["abc xyz"]) == {"cer": 3 / 7, "wer": 0.5}


def test_no_accents_and_empty_predictions():
    assert score(["Đặng"], ["Dang"], without_diacritics=True) == {"cer": 0, "wer": 0}
    assert score(["abc"], [""]) == {"cer": 1, "wer": 1}
    assert normalize(" a  b ", "none") == " a  b "
