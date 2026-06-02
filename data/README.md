# Data Dictionary

## sections_labeled_manually.xlsx

64 manually labelled contract sections, prepared during prior industry
collaboration with the partner firm. Each row is one section from a SaaS or
ERP contract, annotated by the development team after keyword-based candidate
identification and manual semantic comparison against the requirement
catalogue.

| Column                  | Type | Description                                            |
|-------------------------|------|--------------------------------------------------------|
| Unnamed: 0              | int  | original pandas index, ignored                         |
| contract                | int  | source contract identifier                             |
| paragraph               | str  | paragraph marker as found in the contract              |
| section                 | str  | section marker within the paragraph                    |
| section_content         | str  | original section text                                  |
| clean_section_content   | str  | lowercased, punctuation-stripped section text          |
| catalog_id              | int  | gold-standard label, 1-indexed into the catalogue      |

`catalog_id` maps to the catalogue via `catalogue.iloc[catalog_id - 1]`.

Class coverage: 62 of the 76 catalogue classes are represented in the
labelled set. Two classes (44 and 62) have two examples each, all others one.
14 classes have no labelled instance and therefore contribute zero to
per-class metrics.

## catalogue_clean_with_aspects.xlsx

76 requirement entries developed by the partner firm. Each entry describes
one clause topic that the contract review system should detect.

| Column              | Type | Description                                                  |
|---------------------|------|--------------------------------------------------------------|
| paragraph_topic     | str  | broader topical area (e.g. payment terms, liability)         |
| section_topic       | str  | guiding question for the specific topic                      |
| example             | str  | example sentence illustrating the topic                      |
| core_aspects        | str  | aspect list for fulfilment scoring, NOT used in this benchmark |
| prompts ab zeile 46 | str  | legacy column, largely empty, ignored                        |

For the topic classification benchmark, only `paragraph_topic`,
`section_topic`, and `example` are relevant. The exact construction of the
reference text used for cosine similarity is documented in
`notebooks/01_encoder_benchmark.ipynb`.

## Source

Both files are exports from the partner firm's prior prototype repository
(`Masterthesis_Team3_contract_checker`). They have not been altered for this
thesis other than being moved into this repository structure.
