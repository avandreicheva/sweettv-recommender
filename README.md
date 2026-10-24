# sweet.tv TV Program Recommender

Kaggle competition: predict each user's 5 most-preferred `tv_show_id`s for weeks 31-42,
scored by MAP@5 (AP normalized by `min(5, |R_u|)`).

**Final submission: `submission_hybrid_v5.csv`** — private **0.41196**, public **0.42482**
(best of 30+ submissions explored; see `results/kaggle_submission_history.csv`).

## Environment

- Python 3.9.6, macOS-26.5.2-arm64
- pandas 2.3.3, numpy 2.0.2, scipy 1.13.1, pyarrow 21.0.0, kaggle 1.7.4.5
- Install: `pip3 install --user pandas numpy scipy pyarrow kaggle`
- Everything runs as plain scripts, no notebooks, no GPU required. Full pipeline
  (build features -> tune -> build+validate+submit a file) runs in well under a
  minute per stage on a single laptop core.

## Data (not included — download from the Kaggle competition page)

Place these at the repo root before running anything:
- `dataset11-30.csv` — view log, weeks 11-30 (`user_id, vsetv_id, start_time, stop_time, duraton`)
- `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`, `export_arh_31-42-final.csv` — channel schedules
- `submission.csv` — the 1,260-user prediction template (user order + header to match exactly)

## Run order

```
scripts/01_file_profile.py              # rows/cols/time range per file, week numbering
scripts/02_view_log_stats.py            # users, channels, duraton quantiles (streaming)
scripts/03_schedule_stats.py            # tv_show_id!=0 share, categories per schedule file
scripts/04_submission_overlap.py        # submission users vs view log / labelled programmes
scripts/05_catalog_overlap.py           # catalog overlap weeks 31-42 vs 11-30

scripts/06_build_interactions.py        # user x tv_show_id interaction tables, join rule applied
scripts/07_candidate_pools.py           # candidate pools (distinct labelled shows) for VAL/TEST
scripts/08_evaluator.py                 # MAP@5 evaluator (importable module)
scripts/common_eval.py                  # shared data loading / scoring helpers (importable module)

scripts/09_baselines.py                 # popularity + user-history, MAP@5 on VAL
scripts/10_tune_popularity_and_history.py
scripts/11_als.py                       # implicit ALS from scratch (numpy/scipy), grid-tuned
scripts/12_ppr.py                       # personalized PageRank on a show co-view graph
scripts/13_hybrid.py                    # history+PPR fusion, airing-frequency factor
scripts/14_final_submissions.py         # refit all methods on TRAIN_FULL, write Kaggle files
scripts/15_expected_watch_time.py       # EWT ranker (airtime-seconds boost -- later dropped)
scripts/16_submission_hybrid_v2.py

scripts/17_build_loose_interactions.py  # >=60s-only join variant (training features only)
scripts/18_retune_hybrid_v1.py          # selection criterion fixed to mean(ALL,TOP10,TOP20)
scripts/19_submission_hybrid_v3.py      # loose-join hybrid (looked better in validation,
                                         # WORSE on the real leaderboard -- see Findings)

scripts/20_hybrid_v1_variants.py        # one-factor variants around v1 -> found basis=count (v4)
scripts/21_build_variant_interactions.py # count-as-distinct-days + alt join-threshold tables
scripts/22_hybrid_v4_variants.py        # one-factor variants around v4 -> combined into v5

scripts/23_user_case_studies.py         # heavy/light viewer walkthrough for the report
scripts/test_evaluator.py               # evaluator unit tests (hand-worked AP@5 examples)
```

Each script writes its own `results_*.csv` / `interactions_*.parquet` / `submission_*.csv`
to the repo root (gitignored except for the tracked tables under `results/` and the final
`submission_hybrid_v5.csv` — see **Repo layout**). Run logs are kept under `scripts/out/`.

## Pipeline design

- **Join rule**: `view_log.vsetv_id == schedule.channel_id`, matched by ISO week to the
  right schedule file; a view counts for *every* labelled (`tv_show_id != 0`) programme whose
  overlap passes **>=120s AND >=20% of the programme's duration**; multi-programme overlaps
  all count; views on channels absent from the schedule are dropped.
- **Interaction table**: `user_id, tv_show_id, count, watched_seconds, last_watch_time` for
  `TRAIN_A` (weeks 11-20), `VAL` (weeks 21-30, ground truth only, join never loosened), and
  `TRAIN_FULL` (weeks 11-30, final refit).
- **Candidate pool**: distinct labelled shows airing in the *target* window's schedule only
  (never from view logs) — `candidate_pool_VAL.csv` (21-30) / `candidate_pool_TEST.csv` (31-42).
- **Evaluator**: MAP@5, AP normalized by `min(5,|R_u|)`, users with empty `R_u` excluded from
  the mean but counted. Ground truths tried: ALL (everything watched), TOP-5/10/20 preferred
  (by watched_seconds then count). **Selection criterion settled on `mean(ALL, TOP10, TOP20)`**
  after TOP-5-only tuning was shown (via real Kaggle feedback) to mis-rank methods.
- **Methods**: popularity, user-history (recency-weighted), implicit ALS (confidence-weighted,
  implemented from scratch with numpy/scipy — no external recsys library), personalized
  PageRank on a show co-view graph, and a hybrid fusion of history + PPR with an
  airing-frequency boost.

## Final model: `submission_hybrid_v5.csv`

| hyperparameter | value |
|---|---|
| join (TRAIN_FULL features) | strict: overlap >=120s AND >=20% of programme duration |
| history score basis | `count` (interaction count, not watched_seconds) |
| history recency half-life | 28 days |
| PPR alpha | 0.8 |
| PPR restart vector | built from `count` (not watched_seconds) |
| airing-frequency factor | `(1 + log(airings_in_target_schedule)) ** 2`, applied to both fused components |
| fusion weight `w` (history share) | 0.6 |
| fallback for missing slots | popularity (by total watched_seconds, pool-restricted) |
| candidate pool | `export_arh_31-42-final.csv`, distinct `tv_show_id != 0` |
| random seed | 42 (`common_eval.SEED`; used for popularity tie-break jitter — PPR/history/hybrid are otherwise deterministic solves) |

**Runtimes** (single core, laptop): validation fit+score (TRAIN_A -> VAL) ≈ 4.0s; final
refit+rank+write (TRAIN_FULL -> TEST pool, 1,260 users) ≈ 3.8s. The full one-factor sweep that
found this config (11 variants around v1 + 11 around v4) took ≈ 85-110s per round of 11.

## Results

Validation-vs-leaderboard progression (see `results/kaggle_submission_history.csv` for every
submission made):

| version | what changed | private | public |
|---|---|---|---|
| hybrid v1 | history+PPR fusion, basis=both, airing β=1 | 0.36586 | 0.39013 |
| user-history | standalone, basis=both | 0.34974 | 0.37586 |
| hybrid v2 | history replaced by expected-watch-time + airtime-seconds boost | 0.31971 | 0.34210 |
| hybrid v3 | v1 retuned on mean(ALL,TOP10,TOP20), loose join | 0.34634 | 0.35821 |
| PPR (standalone) | alpha=0.8 | 0.29106 | 0.32348 |
| popularity (standalone) | by total watched seconds | 0.29088 | 0.30201 |
| ALS (standalone) | implicit, factors=20, a=0.05 | 0.22518 | 0.24443 |
| **v4** | v1 + `basis=count` only | 0.39304 | 0.41066 |
| **v5 (final)** | v4 + half-life=28, airing β=2, w=0.6, PPR restart=count | **0.41196** | **0.42482** |

## Key findings / pitfalls

- **The TOP-5-only validation truth mis-ranks methods.** It rewarded an airtime-seconds boost
  (hybrid v2) that scored worst on the real leaderboard, and ranked v1 below it. Switching the
  selection criterion to `mean(ALL, TOP10, TOP20)` recovered the correct v1 > user-history order.
- **Even the fixed criterion isn't fully trustworthy for fine hyperparameter choices.** The
  loose join (hybrid v3) and `count = distinct viewing days` both scored *better* in validation
  than their strict-join / event-count counterparts, but *worse* on the real leaderboard. Both
  were discarded in favor of the Kaggle-confirmed winners.
- **basis=count was the single biggest real-world win** (+0.027 private / +0.021 public over
  v1) — bigger than anything found by validation-only tuning.
- **Individually-validated-on-Kaggle factors compound.** Combining half-life=28, airing β=2,
  w=0.6, and PPR-restart=count into v5 beat every individual factor's gain, even though v5's
  *validation* score was lower than v4's — real leaderboard feedback was the only reliable
  signal for this step.
- **Airing-count beta genuinely helps** (number of airings in the target schedule); the
  superficially similar airtime-*seconds* boost (EWT ranker, hybrid v2) does not — the two are
  easy to conflate but behave oppositely on the real metric.

## Case studies & tests

- `scripts/23_user_case_studies.py` walks through one heavy viewer (445 distinct shows,
  7.24M watched seconds in weeks 11-20) and one light viewer (1 show, 2,753s) among the
  submission users: Top-5 from every method in the validation setup, their actual weeks 21-30
  watches, and their real `submission_hybrid_v5.csv` row. For the heavy viewer it also prints
  a hand-checkable AP@5 breakdown (4/5 hits -> AP@5 = 0.8000, confirmed against the evaluator).
- `scripts/test_evaluator.py` unit-tests the evaluator against two hand-worked examples
  (`pred=[A,B,D,C,E]` vs `relevant={A,B,C}` -> AP@5=0.9167; `pred=[D,A,E,B,C]` -> AP@5=0.5333)
  plus empty-`R_u` exclusion and `map_at_k` aggregation. Run: `python3 scripts/test_evaluator.py`.

## Repo layout

```
.
├── README.md
├── .gitignore
├── submission_hybrid_v5.csv     # final submission
├── results/                     # tracked results tables (grid searches, submission history,
│                                 # show id -> title/category lookup used by the case studies)
└── scripts/                     # full pipeline, numbered in run order, + out/ run logs
```

Raw competition data, Kaggle credentials, and all intermediate interaction/candidate-pool/
prediction caches and non-final submission variants are gitignored (fully reproducible by
re-running the scripts above against the downloaded data).
