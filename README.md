# C147/247 Final Project
### Winter 2026 

Specifically, this branch focuses on the CNN + GRU architecture.
Files I changed:
- '''emg2qwerty/modules.py''': added CNNRNNEncoder to replace baseline encoder
- ```emg2qwerty/lightning.py```: added CNNRNNCTCModule to replace TDSConvCTCModule
- config/model/cnn_rnn_ctc.yaml: to replace tds_conv_ctc.yaml

I ran this locally on my own computer:

To train (with default options user="single_user", trainer.accelerator=gpu, trainer.devices=1):
```
python -m emg2qwerty.train model=cnn_rnn_ctc
```

