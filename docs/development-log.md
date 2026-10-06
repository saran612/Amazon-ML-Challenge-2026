# Development log

How the model reached its current form, one step at a time. All scores are macro F0.5 on
validation half B (55,357 held-out S1 records, never used for tuning), on the full data.

| Step | Change | Macro F0.5 | Precision / recall | Notes |
|---|---|---|---|---|
| 0 | Blocking score only | 0.725 | 0.77 / 0.81 | reference line |
| 1 | Full pipeline: 4 blocking passes, 58 features, two-stage LightGBM, calibrated F0.5 decision | 0.9775 | 0.994 / 0.949 | blocking recall 98.27% |
| 2 | More candidates (12 → 16 per record), local name rarity | 0.9777 | 0.994 / 0.949 | blocking recall 98.45%; no real gain |
| 3 | House-number distance, sibling and competitor features; 30M training rows, 255 leaves | 0.9830 | 0.995 / 0.963 | biggest single step |
| 4 | Cross-encoder on the uncertain band (default) | **0.9861** | 0.998 / 0.964 | +0.003 from ~0.7% of pairs |
| 5 | Simulated denser crowd (20% of S1 removed from training) | 0.9850 | 0.996 / 0.965 | no gain; kept as an option |

## 1. First full pipeline

The first complete version already had most of the architecture: generic normalization plus
learned vocabulary, a fine-tuned multilingual encoder, four retrieval passes, 58 features, a
two-stage LightGBM and an expected-F0.5 decision layer. Validation reached **0.9775**, against a
ceiling of 0.994 for the retrieved candidates. The model was precise (0.994) but missed about 5%
of true matches. The record-consistency step, tuned together with "off", chose "off", and has in
every run since.

## 2. More candidates did not help

Blocking missed 3,296 true pairs, the third-largest loss, so the cap went from 12 to 16 candidates
per record and the embedding pass from 10 to 15 neighbours. Recall of blocking rose to 98.45%,
but the matcher could not use most of the extra candidates: **0.9777**, within noise. Features for
local name rarity (same name *in the same place*) barely touched the largest error slice either.

## 3. Reading the errors

The error analysis saves examples of every mistake, and they pointed at house numbers:

| Kind | Record | S1 record |
|---|---|---|
| missed match | cascade committee of albion, **305** h street | **304** h street |
| missed match | jordan madlin, **61** greenleaf drive | **60** greenleaf drive |
| wrong merge | fresh property group, **2026** n dakota avenue | **2005** n dakota avenue |

The data contains noise on true matches (off by one) *and* different businesses with the same
name nearby. The model had no notion of how far apart two numbers are, and compared each record
only with the S1 record. Three feature groups followed:

- **house-number distance**: absolute and relative difference, off-by-≤2, digit edit distance,
  closest pair of numbers;
- **siblings**: how the record compares with the S1 record's other strong records, including
  whether it shares *their* house number;
- **competitors**: whether this S1 record is the best address match among the query's candidates
  with the same name.

With twice the training rows and bigger trees, LightGBM alone reached **0.9830** (+0.0053),
mostly through recall (0.949 → 0.963). The relative house-number difference became the third most
important feature, and sibling agreement and competitor gaps entered the top ten.

## 4. Cross-encoder

A cross-encoder reads both records together, which catches typos and transliterations that
pairwise similarity features miss, but it is expensive. Here it only re-scores the uncertain band
(calibrated probability 0.02–0.98, about 0.7% of pairs) and is blended with LightGBM with a
weight tuned on validation half A. Trained on 200k hard pairs in 15 minutes on the laptop GPU, it
earned weight 0.5 and added **+0.0031**, to **0.9861**. Together, steps 3 and 4 cut wrong merges
from 1,024 to 359 and missed matches from 9,828 to 6,843.

## 5. A denser crowd

The files to be matched contain more S2/S3 records per S1 record than the training files (5.76
vs 4.67). To train and tune in a similar crowd, 20% of the non-validation S1 records were removed
from the index, turning their records into distractors (26% → 40% of all records). Validation
stayed at **0.9850** and the chosen threshold barely moved: the removed businesses' records are
*loose* distractors, while the harder cases are near-copies of businesses that *are* in the
index. The option stays (`--sim-drop`), off by default.

## 6. Transfer to an unseen country

One country appears only in the data to be matched. The optional leave-one-country-out mode
trains on one training country and evaluates on the other:

| | Step 1 | Step 4 |
|---|---|---|
| train US → evaluate India | 0.925 | 0.934 |
| train India → evaluate US | 0.963 | 0.976 |

The drop is mostly precision (wrong merges), worst on businesses with no match. A stricter
minimum probability for countries absent from training was searched on these directions; no
single value improved both, so it is never applied.

## Engineering notes

- **Memory**: training features are written to a memory-mapped file and binned once for all
  LightGBM folds; candidate pairs are written per country and streamed from disk while scoring.
  Peak memory stays around 20 GB for 200M candidate pairs.
- **Caching**: every stage has its own cache key (settings + upstream keys), so changing, say,
  the number of candidates reruns blocking and later stages but keeps the embeddings.
- **OpenMP**: torch, transformers and FAISS run in spawned worker processes; mixing their OpenMP
  runtimes with LightGBM's in one process is fragile on macOS.
- **Speed** (M4 Pro, 24 GB): blocking the training data about 1 h, embedding 11M names about
  35 min, LightGBM about 30 min, scoring 127M validation pairs about 1.4 h.
