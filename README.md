# Melimda: Machine Learning Improved Docking (scoring) Algorithm

Melimda is a simple tool to re-score molecular docking results generated using
Autodock Vina. Melimda is trained using experimentally validated data of molecular
binding derived from the PDBBind database, and employs Morgan Fingerprints and 
Binding Site descriptors, along with the Vina raw score, to estimate a more
accurate energy of interaction. It requires the `.pdbqt` files of the Vina output (`--ligand`) and receptor (`--receptor`).

## Requirements

- python (3.8)
- pdbfixer
- Autodock Vina
- OpenBabel

## Setup

```
$ pip install https+git@github.com:omixlab/melimda-cli.git
```

## Running

```
$ melimda-predict \
    --ligand result.pdbqt  \
    --receptor receptor.pdbqt \
    --model model.pkl \
    --output result.txt > /dev/null
```

## Cite Us

Goulart, L (2025). *Melimda: Machine Learning Improved Docking (scoring) Algorithm*. Available at: https://github.com/omixlab/melimda-cli. 
