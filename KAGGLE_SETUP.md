# Kaggle setup

1. In Kaggle, create a new notebook and select a GPU accelerator (T4 x2 or P100).
2. Create or update a private Kaggle Dataset containing this project source:
   `config.py`, `src/`, `kaggle_notebook.py`, and `requirements.txt`.
   Do not include the large dataset ZIP files in this source dataset.
3. Add that source dataset to the notebook as an input.  The runner locates it
   by finding `config.py` and the `src/` directory.
4. Add the CrackVision12K ZIP as a second notebook input.  Add the CrackForest
   ZIP as an optional third input.  The runner extracts both archives itself.
5. Paste the complete contents of `kaggle_notebook.py` into one notebook code
   cell and run it.  Keep Internet off unless Kaggle reports a missing Python
   package during Step 2.
6. Retrieve `checkpoints/`, `results/`, and `features/` from the notebook
   output after the run completes.

The CrackVision12K archive must retain its `split_dataset_final/train`,
`split_dataset_final/val`, and `split_dataset_final/test` folders, each with
`IMG/` and `GT/` subfolders.
