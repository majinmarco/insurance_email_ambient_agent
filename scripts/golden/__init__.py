"""Golden-dataset build tooling (MAR-9).

``labels`` derives hand-correctable ground-truth labels from the smoke generator's
known structure (``Skeleton`` + ``Scenario`` + ``NoisePlan``). The builder
(``scripts/build_golden_dataset.py``) uses it to freeze committed fixtures under
``tests/golden/`` that the eval suite (MAR-10) loads deterministically.
"""
