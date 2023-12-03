# voco Project

Based on this [template](https://github.com/WrathOfGrapes/asr_project_template) and [this FastSpeech 1 implementation](https://github.com/xcmyz/FastSpeech).

## Installation guide

```
pip install -r ./requirements.txt
gdown  -O default_test_model/checkpoint.pth --fuzzy

```

## Usage guide

You can choose between 2 pipelines: "the clown" and "the normal one".

"The clown" is trained using 2 different configs and fixing dataset bug in between.  
"The normal one" is trained using one config and no bugs, but its checkpoint undertrained at the moment.

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
python3 train.py --config-name hifigan-bug.yaml +trainer.checkpoint_path=default_test_model/checkpoint.pth
```

"The normal one":

```shell
gdown $your_link -O default_test_model/checkpoint.pth --fuzzy
python3 train.py --config-name hifigan.yaml +trainer.checkpoint_path=default_test_model/checkpoint.pth
```

You can change wavs in test_data if you want.