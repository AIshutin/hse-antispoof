# Speech Antispoofing Project

Based on this [template](https://github.com/WrathOfGrapes/asr_project_template)

## Installation guide

```
pip install -r ./requirements.txt
mkdir default_test_model
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
unzip LA.zip
```

## Usage guide

### To train

```shell
python3 train.py --config-name lcnn_cqt ++kaiman=False ++loss.weight=[] ++lr_scheduler.gamma=0.95 ++seed=0
```

### To test:

```shell
gdown https://drive.google.com/file/d/1fBgiJCJ9jO1Ab_J0WKf2097-x8fqRREb/view?usp=sharing -O default_test_model/checkpoint.pth --fuzzy
python3 test.py
```

The data will be logged to wandb. You can change wavs in `default_test_data`` if you want.