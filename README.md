# C147/247 Final Project
### Winter 2026 

Specifically, this branch focuses on the CNN + GRU architecture.
Files I changed:
- ```emg2qwerty/modules.py```: added CNNRNNEncoder to replace baseline encoder
- ```emg2qwerty/lightning.py```: added CNNRNNCTCModule to replace TDSConvCTCModule
- ```config/model/cnn_rnn_ctc.yaml```: to replace tds_conv_ctc.yaml

I ran this locally on my own computer and didn't really use the notebook except to generate some figures.

In a terminal...

Set up the env:
```
git clone https://github.com/RonitBarman/emg2qwerty.git
git checkout -b CNNGRU
git pull origin Katelyn-CNNGRU-Hybrid

cd ./emg2qwerty
conda env create -f environment.yml
conda activate emg2qwerty
pip install -e .
```

Download the single user dataset from this link: 
https://ucla.app.box.com/s/3xc4nwpfjfpo6ydjs94t0v2kuq37d5eg

And place the files in a new top-level ```data``` directory.

To train (with default options user="single_user", trainer.accelerator=gpu, trainer.devices=1):
```
python -m emg2qwerty.train model=cnn_rnn_ctc
```

To test and decode:
```
python -m emg2qwerty.train \
  user="single_user" \
  "checkpoint='best_checkpt_path_found_in_logs'" \
  train=False \
  trainer.accelerator=gpu \
  decoder=ctc_greedy # or ctc_beam
```
