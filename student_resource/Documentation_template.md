# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Arete  
**Team Members:** Joe, Abhinav, Chanchal, Immanuel  
**Submission Date:** 27/09/2026

---

## 1. Executive Summary
Our solution follows a hybrid entity-resolution pipeline: we first reduce the search space with a strong blocking rule built on normalized business names and addresses, then apply a calibrated binary classifier to score candidate pairs. The blocking stage is designed to preserve recall while keeping the candidate set compact, and the matcher focuses on precision by using robust token-level and string-similarity features. This combination is well suited to the noisy, partially inconsistent business records in the challenge data.

---

## 2. Methodology

### 2.1 Problem Analysis
The data is highly noisy and heterogeneous across sources. Business names frequently vary due to abbreviations, legal suffixes, punctuation, transliteration, and formatting inconsistencies, while addresses may differ in road/street naming, missing components, landmark references, or incomplete postal information. There are also country-level differences and partial matches that are not always identical at the string level. Because the challenge scoring is precision-heavy, a purely “loose” blocking strategy would create too many false candidates, while a strict strategy would risk missing valid matches.

### 2.2 Solution Strategy
We used a two-stage pipeline:

1. Candidate generation through normalized name + address blocking
2. Match scoring with a logistic-regression classifier trained on positive and sampled negative pairs

This is a hybrid blocking + classifier approach, designed to balance recall in the first stage with precision in the second stage.

**Approach Type:** Hybrid (Blocking + Classifier)  
**Core Innovation:** A robust normalized-name/address blocking index that suppresses cosmetic noise and a compact feature set that captures token overlap, edit similarity, and country consistency for precision-oriented matching.

---

## 3. Candidate Generation (Blocking)
The blocking stage creates a candidate set for each Source 1 entity by indexing records from Sources 2 and 3 using normalized text forms. We did not rely on exact raw string equality; instead, we normalized name and address fields to reduce common noise patterns such as abbreviations, punctuation, legal suffixes, and address shorthand.

- **Blocking keys used:** normalized business name variants, normalized address strings, source-country consistency checks
- **Candidate pairs generated:** produced by the blocking stage in `output/candidate_pairs.tsv`; every Source 1 entity in the test set is represented once with its candidate list
- **How we ensured true matches were not lost:**
  - business names were normalized by converting common suffixes and abbreviations to canonical forms, stripping legal suffixes, and removing domain-style tokens
  - address strings were normalized to a comparable representation that removes noisy markers like landmarks, country labels, and repeated apartment/unit fragments
  - candidate generation combined both name and address evidence, and pairs were filtered by country compatibility when both country values were available
  - this design keeps the candidate pool broad enough to cover the known match patterns while still reducing the full cross-product dramatically

A key implementation detail is that the candidate set is the last blocking stage before model inference, which matches the challenge requirements for the pipeline audit and validation.

---

## 4. Matching Model

**Features used:**
- Name features:
  - Jaccard similarity on normalized name tokens
  - token overlap between the business names
  - sequence-based similarity on normalized name strings
- Address features:
  - Jaccard similarity on normalized address tokens
  - token overlap between address strings
  - sequence-based similarity on normalized address strings
- Other:
  - country match indicator

**Model type:** Logistic Regression with class balancing  
**Threshold selection method:** candidate pairs are scored by the trained model probability; a default probability threshold of 0.5 is applied to keep the final predictions precision-focused while allowing matches when the feature evidence is strong.

The training pipeline builds pairwise examples from the blocking candidate set, computes the above features for each pair, and trains a balanced logistic regression classifier. This keeps the model small, interpretable, and efficient while still capturing the main discriminative signals in the dataset.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** best validation result is obtained by the trained logistic model on the generated candidate pairs; the pipeline’s outputs are recorded in the project outputs and validated against the challenge submission rules
- **Common false positives (wrong merges):** records with similar names or overlapping address tokens but distinct entities, especially when the same business type or street/location context appears across multiple establishments
- **Common false negatives (missed matches):** records with aggressive formatting differences, partial addresses, or missing/renamed legal identifiers that failed to match the canonical name/address representation even though the pair was otherwise a true match

The main limitation of the current pipeline is that it still depends on textual normalization quality. In difficult cases where the company name is shortened to a local shorthand or the address is only partially present, the model can struggle unless the blocking stage preserves enough candidate evidence.

---

## 6. Conclusion
Our approach is a practical and robust solution to business entity resolution under noisy, multi-source conditions. By combining a carefully engineered blocking strategy with a lightweight but discriminative matching model, we reduce the comparison space substantially while maintaining good precision and recall on the candidate set. The implementation is transparent, reproducible, and aligned with the challenge requirements for both the scoring pipeline and the final submission package.

---

## Appendix

### A. Code Artefacts
The runnable pipeline is organized as follows:

- `src/blocking.py` — normalization helpers, indexing logic, candidate generation, and evaluation utilities
- `ml/features.py` — feature construction for candidate pairs and model training logic
- `ml/predict.py` — prediction pipeline that scores all blocking candidates and writes `matching_results.tsv`
- `generate_candidate_pairs.py` — end-to-end candidate generation for the test data
- `output/` — generated candidate and final match files

To reproduce the outputs:

1. Generate candidate pairs from the test data using the blocking stage
2. Train the matcher on the training set
3. Score every candidate pair with the classifier
4. Write the final `matching_results.tsv` and the candidate file `candidate_pairs.tsv` into the output directory

The project is designed to be executable from the repository root with the provided scripts and the challenge dataset layout.

### B. Additional Results
The blocking and matching outputs are stored in the repository’s `output/` folder, and the files are validated against the challenge submission rules before use. The logic explicitly enforces one row per Source 1 entity, candidate IDs restricted to S2/S3 records, and output formatting in TSV form to satisfy the scorer requirements.

---

**Note:** This methodology reflects the implemented solution in the current pipeline and is tailored to the actual blocking and matching logic used for the challenge submission.
