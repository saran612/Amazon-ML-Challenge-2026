# Methodology

## 1. Problem

Three sources describe overlapping sets of businesses. Source 1 (S1) is deduplicated; sources 2
and 3 (S2, S3) contain noisy, partial copies plus records that match nothing. For each S1 record
the task is to return every S2/S3 record of the same business.

Scoring is **macro F0.5**: F0.5 is computed per S1 record and averaged over all of them.
F0.5 weights precision twice as much as recall, and an S1 record with no true match scores 1.0
only when its prediction is empty. Two consequences shaped the design:

- A wrong merge costs more than a missed match. With 4 true matches, returning 3 correct ones
  scores 0.94; returning all 4 plus one wrong one scores 0.83.
- The output set should be chosen **per S1 record**, not with one global cut-off.

What the training data looks like (2.2M S1, 5.0M S2, 5.3M S3 records):

| Observation | Value | Consequence |
|---|---|---|
| Each S2/S3 record matches at most one S1 record | no exceptions in 7.6M matched records | treat S2/S3 records as queries with one owner |
| True matches per S1 record | mean 3.5; 5.6% have none | find whole groups; keep singletons empty |
| S2/S3 records matching nothing | 26% | a real crowd of distractors |
| Distractors sharing a normalized name with some S1 record | 31% | names alone produce false merges |
| Non-Latin names among true matches | 7.2% | transliteration + cross-script embeddings |
| Website domains used as names | 5.2% | space-free character comparison |
| True matches with an empty address | 4.4% | the name alone must retrieve them |
| Cross-country matches | none | country is a hard partition |

## 2. Normalization and learned vocabulary

Generic steps only: `anyascii` transliteration of non-ASCII text (it must run *before* any
accent stripping, since Unicode decomposition deletes Indic vowel signs), lower-casing,
punctuation removal, digit/letter splitting in addresses, and detection of domain-style names
(`coastaltungsten.com` → `coastaltungsten`).

Everything language-specific is **learned from the data** rather than written by hand:

- **Token variants**, mined per country from training matches: tokens left unmatched on both
  sides of a true pair are candidate variants, kept when they recur, explain a real share of the
  rarer token's occurrences, and pass a language-agnostic similarity test (in-order
  abbreviation, equal consonant skeleton, or Jaro-Winkler ≥ 0.88). Examples found: `rd → road`,
  `c1inic → clinic`, `praivet → private`, `krnatk → karnataka`. A country unseen in training
  only receives the variants all training countries agree on.
- **Filler words**: name tokens present in more than 1% of a country's names (`llc`, `pvt`,
  `limited`, `services`…) are removed from the *core* name; the full name is still compared.
- **Places and place links**: digit-free address segments that recur are places; places that
  co-occur in the same addresses are linked (a city with its region), so two addresses naming
  the same area at different levels still agree.
- **Name rarity**: how many S1 records share a core name in the country, and in the same place.

Unsupervised statistics are computed on the files being processed, so a new country gets its
own filler words and places without any configuration.

## 3. Multilingual name embeddings

`intfloat/multilingual-e5-small` (MIT, 118M parameters) embeds each name in its original
script, so a Devanagari name can land next to its English S1 record. It is first fine-tuned on
up to 300k training pairs (one per S1 record, validation excluded) with a symmetric in-batch
contrastive loss, then used for blocking pass D and as a cosine feature. torch, transformers and
FAISS run in separate worker processes so their OpenMP runtimes never share a process with
LightGBM's.

## 4. Blocking

Per country, each S2/S3 record queries the S1 index with four passes:

| Pass | Signal | Catches |
|---|---|---|
| A | char-trigram TF-IDF of the core name without spaces, top 10 | typos, reordering, domains |
| B | word TF-IDF of the address, top 8 | trade names: different name, same address |
| C | word TF-IDF of name + address, top 10 | generic names separated by address |
| D | nearest names in the embedding space (FAISS IVF), top 15 | cross-script names |

The union is ranked by a combined score and capped at 16 per record, always keeping the top 3 of
every pass. Retrieval ignores index columns present in more than 1% of S1 records (little
signal, most of the cost). Result on validation: **98.45% of true pairs** at 19.3 candidates per
record, a reduction ratio of 0.99998; a perfect matcher on these candidates would score 0.995.

## 5. Features (77, no country feature)

- **Blocking context**: the four cosines, the combined score, its rank and margin to the query's
  best other candidate, each pass's rank, candidate count.
- **Name**: fuzzy ratio, token-set / token-sort / partial ratio on full and core names,
  Jaro-Winkler, space-free and consonant-skeleton comparisons, soft token matches both ways
  (exact, abbreviation, close spelling), first-token match, IDF of the rarest shared word.
- **Address**: token similarities on the full address, places and street, place-link agreement,
  IDF of the rarest shared word, missing-address flags.
- **House numbers**: shared and fuzzy matches, closest pair of numbers, and the distance between
  first numbers (absolute, relative, off-by-≤2, digit edit distance).
- **Rarity**: how common the core name is in the country and in the same place, and how many of
  the query's candidates share this S1 record's name.
- **Siblings**: how the record compares with the S1 record's *other* strong records (records whose
  best candidate it is): best name, address and joint similarity, and house-number agreement.
- **Competitors**: is this S1 record the best address match among the query's candidates, overall
  and among candidates with the same name, and by how much.

There is deliberately no country feature, so the model applies to countries unseen in training.

## 6. Matcher

- **Stage 1**: LightGBM (255 leaves), 3 folds grouped by S1 record, out-of-fold scores for the
  training rows. Features for 30M training pairs are written to a memory-mapped file and binned
  once; the folds use subsets of that dataset, so training fits in 24 GB of RAM.
- **Stage 2**: the stage-1 score, its context within the query (rank, best other candidate,
  margin, number of strong candidates) and the top 24 base features by importance.
- **Calibration**: isotonic regression on validation half A.
- **Cross-encoder**: `multilingual-e5-small` with a classification head, fine-tuned on training
  pairs that stage 1 found hard (out-of-fold score 0.02–0.98), re-scores the uncertain band and
  is blended with LightGBM in logit space. The weight (including 1.0 = off) is tuned on half A.

## 7. Decision

1. **One owner**: each S2/S3 record goes to its most probable S1 record.
2. **Per-S1 set**: the S1 record's records are sorted by probability and the prefix with the
   highest expected F0.5, `1.25·Σp / (0.25·(Σp_all + m) + k)`, is kept; the empty set wins when
   the probability of no match is higher. A single global threshold is the alternative; the
   method and its parameters are chosen on validation half A.
3. **Consistency (optional)**: a record similar to a confidently matched record of the same S1
   can be pulled towards it; tuned together with "off", and chosen "off" in every run so far.

## 8. Validation protocol

5% of the training S1 records are held out, split into half A (calibration, thresholds, every
tuned switch) and half B (reporting only). Blocking runs over the full training data, so the
held-out records face the full crowd; a query counts as a validation query if its true owner or
any of its candidates is a validation S1 record, and training never sees such a query. Token
variants and the encoder's fine-tuning use only pairs outside the validation set.

An optional leave-one-country-out mode (`--proxy`) trains on one country and evaluates on the
other, as an estimate of performance on an unseen country.

## 9. Error analysis

Every run ends with an error analysis on half B: false positives and false negatives by slice,
with the macro F0.5 gain if the slice were fixed. For the default configuration the largest
remaining losses are generic names shared by five or more S1 records (+0.009 if fixed), empty
addresses (+0.007) and pairs blocking never retrieved (+0.006).
