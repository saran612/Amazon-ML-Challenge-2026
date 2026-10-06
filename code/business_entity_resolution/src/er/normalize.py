"""Generic, language-agnostic normalization (no hand-written language knowledge).

Only character-level handling lives here: transliteration to ASCII, case folding, punctuation,
digit/letter boundaries and domain-name detection. Everything word-specific (abbreviations,
legal forms, place names) is learned from the data in `vocab.py`.
"""
from __future__ import annotations

import polars as pl
from anyascii import anyascii

NON_ASCII = r"[^\x00-\x7F]"
# A name that ends in "label.tld" is a web domain; keep the label as the name.
DOMAIN_RE = r"([a-z0-9][a-z0-9\-]{3,})\.[a-z]{2,6}$"


def ascii_fold(s: pl.Series) -> pl.Series:
    """Transliterate every non-ASCII string with anyascii, then lower-case.

    anyascii must run before any accent stripping: Unicode decomposition would otherwise
    delete Indic vowel signs (they are combining marks).
    """
    s = s.fill_null("")
    idx = s.str.contains(NON_ASCII).arg_true()
    if len(idx):
        s = s.clone()
        s.scatter(idx, pl.Series([anyascii(x) for x in s.gather(idx).to_list()], dtype=pl.String))
    return s.str.to_lowercase()


def _clean(e: pl.Expr) -> pl.Expr:
    return e.str.replace_all(r"[^a-z0-9]+", " ").str.replace_all(r"\s+", " ").str.strip_chars()


def _split_digits(e: pl.Expr) -> pl.Expr:
    return e.str.replace_all(r"(\d)([a-z])", "${1} ${2}").str.replace_all(r"([a-z])(\d)", "${1} ${2}")


def normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Add normalized name/address columns. Keeps entity_id, country, source (if present)."""
    name_f = ascii_fold(df["business_name"])
    addr_f = ascii_fold(df["business_address"])
    out = df.with_columns(
        # original-script name for the multilingual embedding model
        name_raw=df["business_name"].fill_null("").str.replace_all(r"\s+", " ").str.strip_chars(),
        nonascii=df["business_name"].str.contains(NON_ASCII),
        _name_f=name_f.str.replace_all(r"^[^a-z0-9]+|[^a-z0-9]+$", ""),
        _addr_f=addr_f,
    )
    out = out.with_columns(domain=pl.col("_name_f").str.extract(DOMAIN_RE, 1))
    out = out.with_columns(
        is_domain=pl.col("domain").is_not_null(),
        name_norm=_clean(
            pl.when(pl.col("domain").is_not_null())
            .then(pl.col("_name_f").str.replace(r"\.[a-z]{2,6}$", ""))
            .otherwise(pl.col("_name_f"))
        ),
        # address segments = comma-separated parts, each cleaned; empty parts dropped
        addr_segs=pl.col("_addr_f").str.split(",").list.eval(
            _clean(_split_digits(_clean(pl.element())))
        ).list.eval(pl.element().filter(pl.element().str.len_chars() > 0)),
    )
    return out.drop("_name_f", "_addr_f", "domain", "business_name", "business_address")
