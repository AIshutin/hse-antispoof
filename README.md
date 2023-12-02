# voco Project

Based on this [template](https://github.com/WrathOfGrapes/asr_project_template) and [this FastSpeech 1 implementation](https://github.com/xcmyz/FastSpeech).

## Installation guide

```
pip install -r ./requirements.txt
gdown  -O default_test_model/checkpoint.pth --fuzzy

```

## Usage guide

To train:
```shell
python3 train.py --config-name fastspeech2
```

To synthesize audio:
```
python3 train.py --config-name inference_fs2 +trainer.checkpoint_path=default_test_model/checkpoint.pth
```

Feel free to change `voco/config/inference_fs2.yaml` to set texts & alphas for synthesis. Alternatively, you can use Hydra CLI features.