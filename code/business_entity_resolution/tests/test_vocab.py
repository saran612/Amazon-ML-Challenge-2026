import polars as pl

from er import vocab
from er.config import Config


def test_abbreviation_test():
    assert vocab.is_abbrev("st", "street")
    assert vocab.is_abbrev("pvt", "private")
    assert not vocab.is_abbrev("services", "solutions")
    assert not vocab.is_abbrev("rd", "street")


def test_token_map_mined_from_pairs():
    a = ["12 main street springfield"] * 40
    b = ["12 main st springfield"] * 40
    df = {"street": 100, "st": 60, "main": 200, "12": 50, "springfield": 80}
    m = vocab.mine_token_map(a, b, df, Config())
    assert m == {"st": "street"}          # rarer spelling -> more frequent spelling


def test_shared_map_keeps_agreement_only():
    per_country = {"US": {"st": "street", "san": "sun"}, "India": {"st": "street"}}
    assert vocab.shared_map(per_country) == {"st": "street"}


def test_first_house_number():
    f = vocab.add_numbers(pl.DataFrame({"nums_str": ["12 45", ""]}))
    assert f["num1"].to_list() == [12, -1]
    assert f["num1_str"].to_list() == ["12", ""]
