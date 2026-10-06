"""A small synthetic dataset with the same layout and kinds of noise as the real data.

Used by the tests and CI, and to try the pipeline without the real data:

    python scripts/make_synthetic.py --out data_synthetic
    python scripts/train.py --data-dir data_synthetic --no-embed

Every S1 business gets 0-6 records spread over sources S2 and S3, each a noisy variant (typos,
abbreviations, reordered words and address parts, legal-suffix changes, domain-style names,
house-number noise, missing addresses). Distractor records are added too: unrelated businesses,
and near-copies of real ones (extra generic word, shifted house number). The test split adds a
country that does not appear in training. Word lists here only generate fake data; the pipeline
itself uses no hand-written vocabulary.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

WORDS = ["apex", "blue", "cedar", "delta", "eagle", "falcon", "granite", "harbor", "iris", "juniper",
         "keystone", "lotus", "maple", "nova", "orchid", "pioneer", "quartz", "river", "summit", "tiger",
         "union", "vertex", "willow", "zenith", "amber", "birch", "coral", "dune", "ember", "fern",
         "glacier", "hazel", "indigo", "jade", "kestrel", "lagoon", "meadow", "nimbus", "onyx", "prairie"]
TRADES = ["consulting", "builders", "logistics", "dental", "foods", "textiles", "motors", "clinic",
          "software", "traders", "exports", "realty", "labs", "studio", "partners", "services"]
GENERIC = ["group", "holdings", "ventures", "global", "center"]
CITIES = ["springfield", "riverton", "lakeside", "fairview", "oakridge", "hillcrest", "brookfield",
          "westport", "northgate", "eastwood", "greenville", "stonebridge"]
COUNTRY = {
    "US": {"suffix": ["Inc", "LLC", "Corp", "Co"],
           "street": [("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Drive", "Dr"), ("Lane", "Ln")],
           "state": [("Texas", "TX"), ("Ohio", "OH"), ("Oregon", "OR"), ("Iowa", "IA")]},
    "India": {"suffix": ["Private Limited", "Pvt Ltd", "LLP", "Limited"],
              "street": [("Road", "Rd"), ("Nagar", "Ngr"), ("Marg", "Mg"), ("Colony", "Col")],
              "state": [("Kerala", "KL"), ("Gujarat", "GJ"), ("Haryana", "HR"), ("Punjab", "PB")]},
    "France": {"suffix": ["SARL", "SAS", "SA"],
               "street": [("Rue", "R."), ("Avenue", "Av."), ("Boulevard", "Bd"), ("Chemin", "Ch.")],
               "state": [("Bretagne", "BRE"), ("Normandie", "NOR"), ("Occitanie", "OCC")]},
}


class Business:
    def __init__(self, rng: random.Random, country: str):
        c = COUNTRY[country]
        self.country = country
        self.words = [rng.choice(WORDS), rng.choice(TRADES)]
        self.suffix = rng.choice(c["suffix"])
        self.number = rng.randint(1, 9999)
        self.street = rng.choice(WORDS).capitalize()
        self.stype = rng.choice(c["street"])
        self.city = rng.choice(CITIES).capitalize()
        self.state = rng.choice(c["state"])

    def name(self) -> str:
        return " ".join(w.capitalize() for w in self.words) + " " + self.suffix

    def address(self) -> str:
        if self.country == "France":
            return f"{self.number} {self.stype[0]} {self.street}, {self.city}, {self.state[0]}"
        return f"{self.number} {self.street} {self.stype[0]}, {self.city}, {self.state[0]}"


def _typo(rng: random.Random, word: str) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1:] if rng.random() < 0.5 else word[:i] + word[i + 1] + word[i] + word[i + 2:]


def noisy_record(rng: random.Random, b: Business, source: int) -> tuple[str, str]:
    """A noisy variant of business b as written by source 2 (upper case, codes) or 3."""
    words = [w.capitalize() for w in b.words]
    if rng.random() < 0.2:
        i = rng.randrange(len(words))
        words[i] = _typo(rng, words[i])
    if rng.random() < 0.15:
        rng.shuffle(words)
    suffix = b.suffix if rng.random() < 0.6 else rng.choice(["", rng.choice(COUNTRY[b.country]["suffix"])])
    name = " ".join(words + ([suffix] if suffix else []))
    if rng.random() < 0.05:
        name = "".join(b.words) + ".com"
    elif rng.random() < 0.1:
        name = rng.choice(["The ", "M/s ", "-- "]) + name

    number = str(b.number)
    r = rng.random()
    if r < 0.1:
        number = "0" + number
    elif r < 0.15:
        number = str(max(1, b.number + rng.choice([-1, 1])))
    stype = b.stype[1] if rng.random() < 0.5 else b.stype[0]
    state = b.state[1] if source == 2 or rng.random() < 0.3 else b.state[0]
    parts = [f"{number} {b.street} {stype}", b.city, state]
    if b.country == "France":
        parts[0] = f"{number} {stype} {b.street}"
    if rng.random() < 0.1:
        parts = parts[:1] + parts[2:]
    if rng.random() < 0.2:
        rng.shuffle(parts)
    address = "" if rng.random() < 0.04 else ", ".join(parts)
    if source == 2:
        name, address = (name.upper() if rng.random() < 0.5 else name), address.upper()
    return name, address


def decoy_of(rng: random.Random, b: Business) -> Business:
    """A different business that looks like b: an extra generic word and a shifted house number."""
    d = Business(rng, b.country)
    d.words = b.words[:1] + [rng.choice(GENERIC)] + b.words[1:]
    d.suffix, d.street, d.stype, d.city, d.state = b.suffix, b.street, b.stype, b.city, b.state
    d.number = b.number + rng.randint(1, 9)
    return d


def make_split(rng: random.Random, countries: list[str], n_s1: int, ids: set) -> tuple[list, list, list, dict]:
    def new_id(prefix: str) -> str:
        while True:
            x = rng.randint(10**6, 10**9)
            if x not in ids:
                ids.add(x)
                return f"{prefix}-{x}"

    s1, s2, s3, truth = [], [], [], {}
    for country in countries:
        for _ in range(n_s1):
            b = Business(rng, country)
            sid = new_id("S1")
            s1.append((sid, b.name(), b.address(), country))
            truth[sid] = []
            n_rec = 0 if rng.random() < 0.06 else rng.choice([1, 2, 3, 3, 4, 4, 5, 6])
            for _ in range(n_rec):
                source = rng.choice([2, 3])
                qid = new_id(f"S{source}")
                (s2 if source == 2 else s3).append((qid, *noisy_record(rng, b, source), country))
                truth[sid].append(qid)
            # distractors: an unrelated business, or a near-copy of this one
            r = rng.random()
            if r < 0.25:
                other = decoy_of(rng, b) if r < 0.12 else Business(rng, country)
                for _ in range(rng.choice([1, 2])):
                    source = rng.choice([2, 3])
                    (s2 if source == 2 else s3).append((new_id(f"S{source}"), *noisy_record(rng, other, source), country))
    return s1, s2, s3, truth


def _write(path: Path, header: list[str], rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_dataset(out_dir: Path, n_s1: int = 300, seed: int = 0) -> None:
    """Write train/ (with ground truth) and test/ in the pipeline's expected layout."""
    rng = random.Random(seed)
    ids: set = set()
    cols = ["entity_id", "business_name", "business_address", "country"]
    for split, countries in (("train", ["US", "India"]), ("test", ["US", "India", "France"])):
        s1, s2, s3, truth = make_split(rng, countries, n_s1, ids)
        rng.shuffle(s2)
        rng.shuffle(s3)
        for k, rows in (("source1", s1), ("source2", s2), ("source3", s3)):
            _write(Path(out_dir) / split / f"{split}_{k}.tsv", cols, rows)
        if split == "train":
            _write(Path(out_dir) / "train" / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"],
                   [(sid, ",".join(q)) for sid, q in truth.items()])
