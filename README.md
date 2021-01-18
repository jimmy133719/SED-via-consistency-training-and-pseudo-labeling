# FP-CRNN via consistency training and pseudo labeling for SED

This work is based on the baseline of DCASE task 4 (https://github.com/turpaultn/dcase20_task4/tree/public_branch/baseline). 

**Note:** Check if the baseline code works before using our code.

<br><br/>
#### Train baseline model
The reproduced baseline model is used to compare with the model trained by proposed methods.
```
python main.py
```
<br><br/>
#### Train audio tagging model 
The model is used for weakly pseudo-labeling strategy.
```
python audio_tagging.py
```
<br><br/>
#### Audio tagging model inference
Inference weak pseudo-label from audio tagging model
```
python audio_tagging_inference.py
```
