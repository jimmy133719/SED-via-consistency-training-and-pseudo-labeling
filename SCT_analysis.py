import glob
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import pdb

from evaluation_measures import psds_score, compute_psds_from_operating_points, compute_metrics
from utilities.utils import generate_tsv_wav_durations

if __name__ == '__main__':
    
    groundtruth_path = # Path of 'validation.tsv'
    durations_path =  # Path of 'validation_duration.tsv'
    # If durations do not exists, audio dir is needed
    groundtruth_audio_path = # Path of 'audio/validation'
    crnn_prediction_path = # Path of crnn w/o SCT prediction
    fpcrnn_prediction_path = # Path of fp-crnn w/o SCT prediction
    crnn_SCT_prediction_path = # Path of crnn w/ SCT prediction
    fpcrnn_SCT_prediction_path = # Path of fp-crnn w/o SCT prediction

    groundtruth = pd.read_csv(groundtruth_path, sep="\t")
    if osp.exists(durations_path):
        meta_dur_df = pd.read_csv(durations_path, sep='\t')
    else:
        meta_dur_df = generate_tsv_wav_durations(groundtruth_audio_path, durations_path)

    long_sound_event = ['Blender', 'Electric_shaver_toothbrush', 'Frying', 'Running_water', 'Vacuum_cleaner']
    groundtruth = groundtruth[groundtruth.event_label.isin(long_sound_event)] 

    # Evaluate crnn w/o SCT
    single_predictions = pd.read_csv(crnn_prediction_path + ".tsv", sep="\t")
    single_predictions = single_predictions[single_predictions.event_label.isin(long_sound_event)]
    crnn_F1 = []
    crnn_PSDS = []
    onset = 0
    while onset < 10:
        gt_temp = groundtruth[(groundtruth.onset >= onset) & (groundtruth.onset < onset + 10/3)]
        pred_temp = single_predictions[(single_predictions.onset >= onset) & (single_predictions.onset < onset + 10/3)]
        meta_dur_temp = meta_dur_df[meta_dur_df.filename.isin(list(set(gt_temp.filename.tolist())))]
        F1, PSDS = compute_metrics(pred_temp, gt_temp, meta_dur_temp)
        crnn_F1.append(F1*100)
        crnn_PSDS.append(PSDS*100)
        onset += 10/3
    
    # Evaluate crnn w/ SCT
    single_predictions = pd.read_csv(crnn_SCT_prediction_path + ".tsv", sep="\t")
    single_predictions = single_predictions[single_predictions.event_label.isin(long_sound_event)]
    crnn_SCT_F1 = []
    crnn_SCT_PSDS = []
    onset = 0
    while onset < 10:
        gt_temp = groundtruth[(groundtruth.onset >= onset) & (groundtruth.onset < onset + 10/3)]
        pred_temp = single_predictions[(single_predictions.onset >= onset) & (single_predictions.onset < onset + 10/3)]
        meta_dur_temp = meta_dur_df[meta_dur_df.filename.isin(list(set(gt_temp.filename.tolist())))]
        F1, PSDS = compute_metrics(pred_temp, gt_temp, meta_dur_temp)
        crnn_SCT_F1.append(F1*100)
        crnn_SCT_PSDS.append(PSDS*100)
        onset += 10/3

    # Evaluate fp-crnn w/o SCT
    single_predictions = pd.read_csv(fpcrnn_SCT_prediction_path + ".tsv", sep="\t")
    single_predictions = single_predictions[single_predictions.event_label.isin(long_sound_event)]
    fpcrnn_F1 = []
    fpcrnn_PSDS = []
    onset = 0
    while onset < 10:
        gt_temp = groundtruth[(groundtruth.onset >= onset) & (groundtruth.onset < onset + 10/3)]
        pred_temp = single_predictions[(single_predictions.onset >= onset) & (single_predictions.onset < onset + 10/3)]
        meta_dur_temp = meta_dur_df[meta_dur_df.filename.isin(list(set(gt_temp.filename.tolist())))]
        F1, PSDS = compute_metrics(pred_temp, gt_temp, meta_dur_temp)
        fpcrnn_F1.append(F1*100)
        fpcrnn_PSDS.append(PSDS*100)
        onset += 10/3

    # Evaluate fp-crnn w/ SCT
    single_predictions = pd.read_csv(fpcrnn_prediction_path + ".tsv", sep="\t")
    single_predictions = single_predictions[single_predictions.event_label.isin(long_sound_event)]
    fpcrnn_SCT_F1 = []
    fpcrnn_SCT_PSDS = []
    onset = 0
    while onset < 10:
        gt_temp = groundtruth[(groundtruth.onset >= onset) & (groundtruth.onset < onset + 10/3)]
        pred_temp = single_predictions[(single_predictions.onset >= onset) & (single_predictions.onset < onset + 10/3)]
        meta_dur_temp = meta_dur_df[meta_dur_df.filename.isin(list(set(gt_temp.filename.tolist())))]
        F1, PSDS = compute_metrics(pred_temp, gt_temp, meta_dur_temp)
        fpcrnn_SCT_F1.append(F1*100)
        fpcrnn_SCT_PSDS.append(PSDS*100)
        onset += 10/3



    # Plot result
    '''
    crnn_F1 = [42.64864832663314, 38.331779331779325, 43.804195804195806]
    crnn_SCT_F1 = [50.39169952506708, 49.75297043382151, 48.713043478260865]
    fpcrnn_F1 = [50.43311316151937, 36.454596022806285, 36.436363636363633]
    fpcrnn_SCT_F1 = [50.44059961749607, 50.8026418026418, 40.062937062937054]
    '''

    x_value = ['0~3.3', '3.3~6.6', '6.6~10']
    plt.plot(x_value, crnn_F1, color='b', label='CRNN')
    plt.plot(x_value, crnn_SCT_F1, color='b', linestyle='dashed', label='CRNN w/ SCT')
    plt.plot(x_value, fpcrnn_F1, color='g', label='FP-CRNN')
    plt.plot(x_value, fpcrnn_SCT_F1, color='g', linestyle='dashed', label='FP-CRNN w/ SCT')
    plt.legend(loc=1)
    plt.ylim(30, 60)
    plt.xlabel('Onset location (s)')
    plt.ylabel('macro F1 score (%)')
    plt.show()
    plt.savefig('SCT_analysis.png')
