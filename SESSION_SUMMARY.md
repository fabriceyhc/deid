# De-identification rebuild — session summary
2026-09-21 / 22

## 1. Why a second pass was needed

The provider notes had already been de-identified upstream (`___` markers), but
measured against the `IP_PATIENT_ID -> PAT_NAME` crosswalk in
`Patient_Identifiers.csv`, that pass leaked systematically:

| Residual PHI in the delivered notes | 2026 extract | 2025 extract |
|---|---|---|
| Note lines with >=1 token of the patient's own name | 8.42% | 6.48% |
| Note lines with >=2 tokens | 5.47% | — |
| Street addresses / ZIPs / pager numbers | 2.6% / 6.1% / 1.8% | — |
| `CREATE_BY` column | full provider name, cleartext, every row | same |

The failures were structural, not random: template headers had the MRN and DOB
stripped and the name left in place (`PATIENT: Jeremy ███  MRN: ___  DOB: ___`),
and hyphenated surnames were half-redacted (`___-Prakash, MD`).

## 2. What was built — `deid2` in /data2/fabricehc/deid

Four layers, precision-descending, arbitrated in `spans.resolve()`:

1. **Self-referential gazetteer** — the study already holds a patient -> name/DOB
   crosswalk, so identifiers are *looked up*, not inferred. Nicknames, initials,
   possessives, hyphen halves, eight DOB formats. Never vetoed. Contributed
   163,363 spans across both corpora.
2. **Roster gazetteer** — 3,988 surnames / 3,599 given names harvested from the
   notes' own `CREATE_BY` column. No external list.
3. **Context rules** — anchored on the measured failure modes above, plus formats
   (phone, email, SSN, address, ZIP, month-name dates, pager/extension, ages 90+).
4. **Transformer NER** — `obi/deid_roberta_i2b2` over overlapping 512-token windows.

Precision comes from two allowlist tiers **derived from the study's own data**
(controlled vocabularies + a corpus document-frequency pass), with every cohort
and roster name subtracted first. A token spanning >=2% of patients cannot be a
patient identifier.

`CREATE_BY` is replaced by a stable `PROV_<hash>` that still works as a grouping key.

## 3. Results

### Corpus scrub
| | 2026.01.22 | 2025.01.10 |
|---|---|---|
| Notes processed | 144,391 | 117,767 |
| Spans redacted | 1,426,455 | 984,514 |
| Runtime (1 GPU) | 2h25m | 2h58m |

**Combined: 2,410,969 spans across 262,158 notes.**

### Final audit — 20 files, ~3.9M rows scanned
- **1** patient-name leak total (a first name in narrative)
- **4** format-probe hits (2 clinic street addresses, double-counted across two files)
- 2026 corpus: 0 and 0. `test/`: 0 and 0.

### Validation (no manual annotation required)
- **Crosswalk recall** — ground truth from the patient->name crosswalk:
  8.42% -> 0.000% across 18,709 patients.
- **Injection benchmark** — 97,380 labelled gold spans injected into real notes:
  **99.88% span-exact**; DATE/PHONE/ADDRESS/ZIP/ID/AGE/EMAIL all 100.00%,
  NAME 99.75%.
- **Clinical-term preservation** — 122,559 occurrences over 29,760 real notes:
  **zero loss** for fentanyl, methamphetamine, methadone, buprenorphine,
  naloxone, heroin, cocaine, cannabis, benzodiazepines and every syndrome and
  antibiotic term tracked.
- 48 regression tests.

## 4. Data-quality issues found along the way

- **`Provider_Notes_sentences.csv` had a 32% duplication defect** — overlapping
  fragments (`'MRN:'` then `'MRN: ___ Date of'`). 22.5M rows where correct
  segmentation gives 10.4M. Your last all9 run spent ~1/3 of its inference on
  duplicated text. Rebuilt clean; a future re-run costs roughly half.
- **Line-level and note-level files disagreed on 1.4% of notes.** Fixed; the
  note file is now built by joining the masked lines, so `join(lines) == full`
  holds by construction.
- **The chunker ran 11x more work than needed.** `en_core_web_sm` with
  `disable=[parser, ner, lemmatizer, textcat]` still runs tok2vec, tagger and
  attribute_ruler, none of which the rule-based sentencizer uses. Switched to a
  blank pipeline — verified byte-identical output, 11 hours -> 19 minutes.

## 5. PHI exposure in git — remediated

Four files with clinical note text were committed and pushed to the public
remote. A full-history scan found **17 such paths**, including training data that
embeds note text inside prompts (no `NOTE_TEXT` column to grep for) and a 47 MB
medical-examiner extract with 8,511 decedent names.

- All 17 purged from every commit with `git filter-repo` (179 -> 177 commits).
- Verified: 0 blobs reachable, `git fsck` clean, fresh own-name sweep across all
  309 tracked data files clean.
- Force-pushed; repo is now **private**.
- `.gitignore` now matches by path family rather than column name — that is what
  let the training `.jsonl` files through.
- Backup mirror: `/data2/fabricehc/sub-co-use-backup-20260921-203340.git`
- Local copies: `/data2/fabricehc/sub-co-use-phi-local-20260921-204719`

## 6. Model bake-off — six configurations, identical 38,965 gold spans

| Config | NAME recall | Missed |
|---|---|---|
| `obi` (current) | 99.79% | 24 |
| `obi + stanford` | 99.79% | 24 |
| **`obi + gliner`** | **99.85%** | **20** |
| `gliner` alone | 92.87% | 679 |
| `ClinicNote-DeID-Baseline` | could not load | — |

- **The newest "medical deid model" on HuggingFace is a stub.**
  `SOTAagi2030/ClinicNote-DeID-Baseline` has 76 bytes of text where the weights
  should be, a 6-word tokenizer, and a README claiming F1 0.921.
- **A second BERT-family model adds nothing** — `obi + stanford` is bit-identical
  to obi alone. Both are i2b2-trained; they share blind spots exactly.
- **Only the different architecture helped**, and only slightly: +0.06pp for 2x
  the compute. Not adopted for the production corpus; available via
  `--gliner-model` for future extracts.

## 7. Outstanding — for you

1. **GitHub Support ticket** for 6 LFS objects. `filter-repo` removes pointers,
   not LFS storage. OIDs in `lfs_phi_oids.txt`; `note_micro_audit.csv`
   (116 patient names) is there in two revisions. Not urgent now the repo is private.
2. **Collaborators must re-clone** — a `git pull` will try to merge the old history back.
3. **IRB-22-1273 notification** — data was public from 2026-02-24.
4. **Gitignored clinician review files still leak** — `exposure_review/units.csv`
   (114/1,141), `judge_adjudication_input.csv` (39/520),
   `endocarditis_review_18.csv` (8/18). Never exposed, but they are the files
   clinicians open. I left the `*_annotated*` / `*_labelled*` ones alone since
   rewriting them risks damaging annotation work — your call.
5. **One known residual**: a patient first name in narrative where the surname was
   already masked (`___ Helene ___`). Space-separated, so the comma-anchored
   repair rule does not fire. 1 instance in 400,000 rows.
6. **`deid` repo is committed but not pushed.** `provider_roster.txt` (7,188 real
   staff names) is gitignored and must stay that way.
