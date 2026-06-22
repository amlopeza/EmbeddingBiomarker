# Data use and licensing

The **code** in this repository is licensed under Apache-2.0 (see `LICENSE`).
This is **independent** of the licensing of the **data** the code processes.

This repository does **not** contain or redistribute any patient-level data.
Everything under `data/` is gitignored; only aggregate result summaries
(`results/*.json`) are versioned. The datasets below must be obtained by the
user from their original sources, under their original terms.

## Data sources and their terms

| Dataset | Source | License | Commercial use |
|---|---|---|---|
| MSK-CHORD 2024 | cBioPortal (`msk_chord_2024`) | **CC BY-NC-ND 4.0** | Not permitted |
| AACR GENIE BPC | Synapse (`syn7222066`) | **CC BY-NC** (verify the exact variant, incl. ShareAlike, in the GENIE Data Guide you accept on Synapse) | Not permitted |

## What this means for users of this repository

- **Non-commercial only.** Both datasets prohibit commercial use. Any model,
  embedding, or result *derived from these data* (including the embeddings and
  the fitted survival head) inherits that restriction in practice. The Apache-2.0
  code license does **not**, and cannot, grant any commercial right over the data.
- **No data redistribution (NoDerivatives, MSK-CHORD).** Do not republish the
  MSK-CHORD dataset or a modified version of it. Aggregate, non-identifiable
  summary statistics (as in `results/`) are not patient-level data; do not commit
  or share patient-level tables, prompts, or per-patient embeddings.
- **ShareAlike (GENIE), if applicable.** If your accepted GENIE terms include
  ShareAlike, any *adaptation of the GENIE data* you distribute must carry the
  same license. This does not affect this independent code.
- **Attribution.** Cite both sources (see below) in any work using this pipeline.
- **Leakage control.** External validation drops MSK-center patients from GENIE
  (MSK also contributes to MSK-CHORD); this is enforced in `external.py`.

## Required citations

- MSK-CHORD: Jee J. et al., *Automated real-world data integration improves
  cancer outcome prediction*, **Nature** (2024).
- AACR Project GENIE: AACR Project GENIE Consortium, *AACR Project GENIE:
  Powering Precision Medicine through an International Consortium*,
  **Cancer Discovery** (2017); plus the specific BPC release citation from its
  Data Guide.

By using this repository together with these datasets, you agree to comply with
each dataset's terms in addition to the Apache-2.0 license of the code.

---

> **Before committing:** confirm in the GENIE Data Guide whether the variant is
> CC BY-NC or CC BY-NC-SA, and adjust the table accordingly. What is certain is
> that it is non-commercial. Also verify the current commercial-licensing contact
> for MSK-CHORD if you intend to mention one.
