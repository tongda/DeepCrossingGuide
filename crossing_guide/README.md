# Documentation

## Usage for Training

1. Prepare Data
2. Run training
3. Export model

### Prepare data

```bash
python preprocess.py --root ./data-landscape --output landscape.csv
```

### Train

```bash
python train.py --data_dir=./data-landscape --save_path=model-land-part-feat-0613.h5 --use_lpf=False --batch_size=64 --num_epoch=30 --all_feat=False --piece_file=./landscape.csv --process_pool_size=10 --orientation=landscape
```

### Export model

```bash
python export.py --path model-land-part-feat-0612.h5 --output-dir weights-landscape-0612
```