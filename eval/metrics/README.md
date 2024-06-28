# Run Metrics

## CLIP, ImageReward

### Installation

```shell
pip install image-reward clip
```

### Compute CLIP & ImageReward
Saves to **results.csv**.

```python
python3 score.py --csv_path [GEN_CSV_PATH]
```

### To only compute CLIP

```python
python3 score.py --csv_path [GEN_CSV_PATH] --score_fns clip
```

### To only compute ImageReward

```python
python3 score.py --csv_path [GEN_CSV_PATH] --score_fns imagereward
```

## HPSv2.0

### Installation

```shell
git clone https://github.com/tgxs002/HPSv2.git
cp img_score.py HPSv2/hpsv2
cd HPSv2
pip install -e .
cd hpsv2
```

### Run HPSv2.0

Saves hpsv2_results.csv here.

```python
python3 img_score.py --csv_path [GEN_CSV_path]
```

## Pickscore & LAION-Aesthetics
See [scorer.py](.../scorer.py)
