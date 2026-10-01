# MultiplePFAnalysis

## Directory layout

- `cache/preprocessed/`: cached data used as analysis input.
- `results/`: generated figures, tables, arrays, and other analysis outputs.
- `1_place_cell_stacks/`: stack and ratemap-correlation analyses.
- `2_place_cell_remapping/`: place-cell remapping analyses.
- `general/`: shared preprocessing, exploratory notebooks, and archived analyses.

Each project-part directory contains its scripts at the directory root and its
notebooks under `notebooks/`.

Keep generated analysis outputs out of `cache/`; that directory is reserved for
data that is loaded by analyses.
