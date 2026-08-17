# Environment records

This directory contains the two complete `pip freeze` records that were retained from the paper
workflow:

- `requirements_finetuning.txt`: PMTransformer fine-tuning and neural-network inference.
- `requirements_analysis.txt`: general SOAP, nomination, and figure-analysis environment.

These files document those two environments; they are not a single universal lock file for every
stage of the project. In particular, the production SMOTE--ExtraTrees classifier used
scikit-learn 1.6.0, as reported in the paper's Supplementary Information, and a separate complete
freeze of that classifier environment was not retained. The analysis freeze contains
scikit-learn 1.6.1, while the fine-tuning freeze contains packages that were present in the
fine-tuning environment but were not used to fit the production classifier.

The `requirements.txt` files under `screening/` and `generation/` are convenience installation
lists with version ranges. They should not be interpreted as records of the exact paper runtime.
