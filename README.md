test

# FP-CRNN via consistency training and pseudo labeling for SED
A sound event detection system designed for DCASE 2020 task 4, which consists of large amount of weak label and unlabel audio clips.

This work is based on the baseline of DCASE task 4 (https://github.com/turpaultn/dcase20_task4/tree/public_branch/baseline). 

**Note:** Check if the baseline code works before using our code.

-------------------------------
### Procedure
#### Train baseline model (optional)
The reproduced baseline model is used to compare with the model trained by proposed methods.
```
python main.py
```
#### Train audio tagging model 
The model is used for weakly pseudo-labeling strategy.
```
python audio_tagging.py
```
#### Audio tagging model inference
Inference weak pseudo-label from audio tagging model
```
python audio_tagging_inference.py -m=model_path -g=../dataset/metadata/validation/validation.tsv
```
Substitude the generated pseudo-label for ../dataset/metadata/train/unlabel_in_domain.tsv
#### Train SED model with ICT+SCT+weak pseudo-label
The model is trained with proposed training strategies, and a novel network architecture, FP-CRNN can be selected to replace CRNN.
```
python main_ISP.py -fpn=True
```
Weak pseudo-label is required, else, replace line 201 with below code
```
weak_class_loss = class_criterion(weak_pred[mask_weak], target_weak[mask_weak])
```
#### Evaluate well-trained model
Choose a well-trained model and evaluate
```
python TestModel.py -m=model_path -g=../dataset/metadata/validation/validation.tsv -fpn=T -lp=F
```
-------------------------------
### Performance
| Method | F1 | PSDS |
| ----- | ----- | --- |
| CRNN (baseline) | 34.8 | 60.0 |
| CRNN w/ISP | 45.1 | 65.8 |
| FP-CRNN w/ISP | 44.5 | 66.9 |
