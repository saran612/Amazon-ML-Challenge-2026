import polars as pl

from er.normalize import ascii_fold, normalize_frame


def test_indic_vowels_survive_transliteration():
    # decomposing first would strip the vowel signs; anyascii must run first
    out = ascii_fold(pl.Series(["राम मार्केटिंग"]))[0]
    assert out.startswith("ram") and "mark" in out


def test_accents_folded():
    assert ascii_fold(pl.Series(["Établissements Cyclo"]))[0] == "etablissements cyclo"


def test_domain_names_and_address_digits():
    df = pl.DataFrame({"entity_id": ["S1-1"], "business_name": ["coastaltungsten.com"],
                       "business_address": ["234D Dallas Ct, Baltimore"], "country": ["US"]})
    out = normalize_frame(df)
    assert out["name_norm"][0] == "coastaltungsten"
    assert out["is_domain"][0]
    assert out["addr_segs"][0].to_list() == ["234 d dallas ct", "baltimore"]
