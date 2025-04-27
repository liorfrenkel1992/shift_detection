Automatic Detection of Domain Shifts in Speech Enhancement Systems Using Confidence-Based Metrics
==============================

An original implementation of the paper: "Automatic Detection of Domain Shifts in Speech Enhancement Systems Using Confidence-Based Metrics, Frenkel et al., ICASSP 2025".

Link to paper:
https://ieeexplore.ieee.org/abstract/document/10888825

Requirements
------------

To run this project, you'll need to install Python packages using:

```bash
pip install -r requirements.txt
```

All models' checkpoints and data are available here:

https://www.dropbox.com/scl/fo/mdtolpxn3jf37311nitlw/AEsCQHK0EAVRPuqJSpvDGWo?rlkey=sskg3pnwh1ppq5h77hxof7zk4&st=x1ejei77&dl=0

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

If you use our code in a scientific publication, we would appreciate using the following citation:

```bibtex
@inproceedings{frankel2025automatic,
  title={Automatic Detection of Domain Shifts in Speech Enhancement Systems Using Confidence-Based Metrics},
  author={Frankel, Lior and Chazan, Shlomo E and Goldberger, Jacob},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}


