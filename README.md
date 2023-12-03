# Vocoder Project

Based on this [template](https://github.com/WrathOfGrapes/asr_project_template)

## Installation guide

```
pip install -r ./requirements.txt
mkdir default_test_model
```

## Usage guide

You can choose between 2 pipelines: "the clown" and "the normal one".

"The clown" is trained using 2 different configs and fixing dataset bug in between.  
"The normal one" is trained using one config and no bugs, but its checkpoint is undertrained at the moment.

### To train

To train "the clown":
```shell
python3 train.py --config-name hifigan-bug
python3 train.py --config-name hifigan2 +resume=saved/hifigan-mk1/path2checkoint.pth
```

To train "the normal one":
```shell
python3 train.py --config-name hifigan
```

### To synthesize audio:

"The clown": 

```shell
gdown https://drive.google.com/file/d/1SoDH__65dA808Eh5EVEJTgcMLgLsldyP/view?usp=sharing -O default_test_model/checkpoint.pth --fuzzy
python3 train.py --config-name hifigan_bug +resume=default_test_model/checkpoint.pth ++trainer.len_epoch=1 ++g_optimizer.lr=0 ++trainer.epochs=77 +data.val.dataset.limit=1
```

"The normal one":

```shell
gdown https://drive.google.com/file/d/1LmNT_XSEHxnd6IgAhX7CULXZEWgFI7Sw/view?usp=sharing -O default_test_model/checkpoint.pth --fuzzy
python3 train.py --config-name hifigan +resume=default_test_model/checkpoint.pth ++trainer.len_epoch=1 ++g_optimizer.lr=0 ++trainer.epochs=77 +data.val.dataset.limit=1
```

The data will be logged to wandb. You can change wavs in test_data if you want.