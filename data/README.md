# Data

The raw data is **not committed to this repository** (it is ~1 GB and
redistributed under Kaggle's terms). To reproduce the analysis, download the
[MLB Game Data](https://www.kaggle.com/datasets/josephvm/mlb-game-data)
dataset from Kaggle and place these three files in `data/raw/`:

```
data/raw/
├── games.csv           # one row per game: scores, records, metadata
├── hittersByGame.csv   # one row per hitter per game
└── pitchersByGame.csv  # one row per pitcher per game
```

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) installed and
configured:

```bash
kaggle datasets download josephvm/mlb-game-data -p data/raw --unzip
```

The dataset covers MLB games from April 2016 through October 2021. Only the
three files above are used; the other files in the dataset (pitches, plays,
events, ...) can be deleted to save disk space.
