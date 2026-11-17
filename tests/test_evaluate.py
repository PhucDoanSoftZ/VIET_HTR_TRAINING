from htrtrain.evaluate import diagnostic_counts, normalize, score


def test_cer_wer_and_unicode_normalization():
    assert normalize(" Tiếng   Việt ") == "Tiếng Việt"
    assert score(["abc def"], ["abc xyz"]) == {"cer": 3 / 7, "wer": 0.5}


def test_no_accents_and_empty_predictions():
    assert score(["Đặng"], ["Dang"], without_diacritics=True) == {"cer": 0, "wer": 0}
    assert score(["abc"], [""]) == {"cer": 1, "wer": 1}
    assert normalize(" a  b ", "none") == " a  b "


def test_digit_and_annotated_proper_name_diagnostics():
    report = diagnostic_counts("Ông An nhận 123 m²", "Ông An nhận 124 m²", ("Ông An",))
    assert report["has_digits"] is True
    assert report["digit_sequence_exact"] is False
    assert report["proper_name_exact_count"] == 1
    assert report["substitutions"] == 1
    assert diagnostic_counts("Tên Lan", "Tên Lan")["proper_name_count"] == 0
