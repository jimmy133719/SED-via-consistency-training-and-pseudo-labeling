# -*- coding: utf-8 -*-
import argparse
import os.path as osp

import torch
# from psds_eval import PSDSEval
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from data_utils.DataLoad import DataLoadDf
from data_utils.Desed import DESED
# from evaluation_measures import compute_sed_eval_metrics, psds_score, get_predictions
from evaluation_measures import psds_score, compute_psds_from_operating_points, compute_metrics, get_f_measure_by_class
from utilities.utils import to_cuda_if_available, generate_tsv_wav_durations, meta_path_to_audio_dir
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms
from utilities.Logger import create_logger
from utilities.Scaler import Scaler, ScalerPerAudio
from models.CRNN_FPN import CRNN_fpn
from models.CRNN import CRNN
import config as cfg
import pdb
from dcase_util.data import ProbabilityEncoder
import scipy
import os

logger = create_logger(__name__)
torch.manual_seed(2020)


def _load_crnn(state, model_name="model",use_fpn=False):
    crnn_args = state[model_name]["args"]
    crnn_kwargs = state[model_name]["kwargs"]
    if use_fpn:
        crnn = CRNN_fpn(*crnn_args, **crnn_kwargs)
    else:
        crnn = CRNN(*crnn_args, **crnn_kwargs)
    crnn.load_state_dict(state[model_name]["state_dict"])
    crnn.eval()
    crnn = to_cuda_if_available(crnn)
    logger.info("Model loaded at epoch: {}".format(state["epoch"]))
    logger.info(crnn)
    return crnn


def _load_scaler(state):
    scaler_state = state["scaler"]
    type_sc = scaler_state["type"]
    if type_sc == "ScalerPerAudio":
        scaler = ScalerPerAudio(*scaler_state["args"])
    elif type_sc == "Scaler":
        scaler = Scaler()
    else:
        raise NotImplementedError("Not the right type of Scaler has been saved in state")
    scaler.load_state_dict(state["scaler"]["state_dict"])
    return scaler

def _load_state_vars(state, gtruth_df, median_win=None, use_fpn=False):
    pred_df = gtruth_df.copy()
    # Define dataloader
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    scaler = _load_scaler(state)
    crnn = _load_crnn(state, use_fpn=use_fpn)
    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler, add_axis=0)

    strong_dataload = DataLoadDf(pred_df, many_hot_encoder.encode_strong_df, transforms_valid, return_indexes=True)
    strong_dataloader_ind = DataLoader(strong_dataload, batch_size=cfg.batch_size, drop_last=False)

    # weak dataloader
    weak_dataload = DataLoadDf(pred_df, many_hot_encoder.encode_weak, transforms_valid, return_indexes=True)
    weak_dataloader_ind = DataLoader(weak_dataload, batch_size=cfg.batch_size, drop_last=False)

    pooling_time_ratio = state["pooling_time_ratio"]
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    if median_win is None:
        median_win = state["median_window"]
    return {
        "model": crnn,
        "strong_dataloader": strong_dataloader_ind,
        "weak_dataloader": weak_dataloader_ind,
        "pooling_time_ratio": pooling_time_ratio,
        "many_hot_encoder": many_hot_encoder,
        "median_window": median_win
    }
def get_predictions(model_list, gt_df_feat, save_predictions, thresholds=[0.5], median_window=None, learned_post=False, use_fpn=False):
    pred_list = []
    # weak if use fpn
    if use_fpn:
        pred_weak_list = []
    for item in model_list:
        model_path = osp.join('stored_data', item, 'model/baseline_best')
        # Model
        expe_state = torch.load(model_path, map_location="cpu")
        params = _load_state_vars(expe_state, gt_df_feat, median_window, use_fpn=use_fpn)

        # Get each model's output
        dataloader = params["strong_dataloader"]
        model = params["model"]
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        pred_strong_all = []
        # weak if use fpn
        if use_fpn:
            pred_weak_all = []
        for i, ((input_data, _), indexes) in enumerate(dataloader):
            indexes = indexes.numpy()
            input_data = to_cuda_if_available(input_data)
            with torch.no_grad():
                pred_strong, pred_weak = model(input_data)
                # _, pred_weak, pred_strong = model(input_data)
            pred_strong = pred_strong.cpu()
            pred_strong = pred_strong.detach().numpy()
            pred_strong_all.append(pred_strong)
            # weak if use fpn
            if use_fpn:
                pred_weak = pred_weak.cpu()
                pred_weak = pred_weak.detach().numpy()
                pred_weak_all.append(pred_weak)            
            if i == 0:
                logger.debug(pred_strong)
        pred_strong_all = np.vstack(pred_strong_all)
        pred_list.append(pred_strong_all)
        # weak if use fpn
        if use_fpn:
            pred_weak_all = np.vstack(pred_weak_all)
            pred_weak_list.append(pred_weak_all)
        del model
    # average pred, still need to deal with output with different shape
    try: 
        pred_strong = sum(pred_list) / len(pred_list)
        # weak if use fpn
        if use_fpn:
            pred_weak = sum(pred_weak_list) / len(pred_weak_list)
            check = (pred_weak > 0.5).astype(float)
            check = np.expand_dims(check, axis=1)
            check = np.repeat(check, 157, axis=1)
            pred_strong = pred_strong * check
    except ValueError:
        print('prediction output needs to be same shape')
    
    # Init a dataframe per threshold
    prediction_dfs = {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()

    # Post processing and put predictions in a dataframe
    decoder = params["many_hot_encoder"].decode_strong
    pooling_time_ratio = params["pooling_time_ratio"]
    median_window=params["median_window"]
    for j, pred_strong_it in enumerate(pred_strong):
        for threshold in thresholds:
            pred_strong_bin = ProbabilityEncoder().binarization(pred_strong_it,
                                                                binarization_type="global_threshold",
                                                                threshold=threshold)
            if learned_post:                
                # adaptive median window
                pred_strong_m = []
                for mw_index in range(len(cfg.median_window)):
                    pred_strong_m.append(scipy.ndimage.filters.median_filter(np.expand_dims(pred_strong_bin[:,mw_index], axis=-1), (cfg.median_window[mw_index], 1)))
                pred_strong_m = np.hstack(pred_strong_m)
            else:
                # fixed median window
                pred_strong_m = scipy.ndimage.filters.median_filter(pred_strong_bin, (median_window, 1))

            
            pred = decoder(pred_strong_m)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            # Put them in seconds
            pred.loc[:, ["onset", "offset"]] *= pooling_time_ratio / (cfg.sample_rate / cfg.hop_size)
            pred.loc[:, ["onset", "offset"]] = pred[["onset", "offset"]].clip(0, cfg.max_len_seconds)

            # pdb.set_trace()
            pred["filename"] = dataloader.dataset.filenames.iloc[j]
            # pred["filename"] = dataloader.dataset.filenames.iloc[indexes[j]]
            prediction_dfs[threshold] = prediction_dfs[threshold].append(pred, ignore_index=True)

            if i == 0 and j == 0:
                logger.debug("predictions: \n{}".format(pred))
                logger.debug("predictions strong: \n{}".format(pred_strong_it))

    # Save predictions
    if save_predictions is not None:
        if isinstance(save_predictions, str):
            if len(thresholds) == 1:
                save_predictions = [save_predictions]
            else:
                base, ext = osp.splitext(save_predictions)
                save_predictions = [osp.join(base, f"{threshold:.3f}{ext}") for threshold in thresholds]
        else:
            assert len(save_predictions) == len(thresholds), \
                f"There should be a prediction file per threshold: len predictions: {len(save_predictions)}\n" \
                f"len thresholds: {len(thresholds)}"
            save_predictions = save_predictions

        for ind, threshold in enumerate(thresholds):
            dir_to_create = osp.dirname(save_predictions[ind])
            if dir_to_create != "":
                os.makedirs(dir_to_create, exist_ok=True)
                if ind % 10 == 0:
                    logger.info(f"Saving predictions at: {save_predictions[ind]}. {ind + 1} / {len(thresholds)}")
                prediction_dfs[threshold].to_csv(save_predictions[ind], index=False, sep="\t", float_format="%.3f")

    list_predictions = []
    for key in prediction_dfs:
        list_predictions.append(prediction_dfs[key])

    if len(list_predictions) == 1:
        list_predictions = list_predictions[0]
    
    return list_predictions, params

def get_variables(args):
    gt_fname, ext = osp.splitext(args.groundtruth_tsv)
    gt_audio_pth = args.groundtruth_audio_dir
    if args.use_fpn=='T':
        use_fpn = True
    elif args.use_fpn=='F':
        use_fpn = False

    if args.learned_post=='T':    
        learned_post = True
    elif args.learned_post=='F':
        learned_post = False

    if gt_audio_pth is None:
        gt_audio_pth = meta_path_to_audio_dir(gt_fname)
        # Useful because of the data format
        if "validation" in gt_audio_pth:
            gt_audio_pth = osp.dirname(gt_audio_pth)

    groundtruth = pd.read_csv(args.groundtruth_tsv, sep="\t")

    return gt_audio_pth, groundtruth, use_fpn, learned_post

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", '--groundtruth_tsv', type=str, required=True,
                        help="Path of the groundtruth tsv file")

    # Not required after that, but recommended to defined
    parser.add_argument("-mw", "--median_window", type=int, default=None,
                        help="Nb of frames for the median window, "
                             "if None the one defined for testing after training is used")

    # Next groundtruth variable could be ommited if same organization than DESED dataset
    parser.add_argument('--meta_gt', type=str, default=None,
                        help="Path of the groundtruth description of feat_filenames and durations")
    parser.add_argument("-ga", '--groundtruth_audio_dir', type=str, default=None,
                        help="Path of the groundtruth filename, (see in config, at dataset folder)")
    parser.add_argument("-s", '--save_predictions_path', type=str, default=None,
                        help="Path for the predictions to be saved (if needed)")

    # Dev
    parser.add_argument("-n", '--nb_files', type=int, default=None,
                        help="Number of files to be used. Useful when testing on small number of files.")

    # Use fpn
    parser.add_argument("-fpn", '--use_fpn', type=str, default='T',
                    help="Whether to use CRNN_fpn architecture, must be same as the saved model.'T' for True, 'F' for False.")
    # Use adaptive post processing
    parser.add_argument("-lp", '--learned_post', type=str, default='T',
                    help="Whether to use adaptive post processing.'T' for True, 'F' for False.")    


    f_args = parser.parse_args()
    # Get variables from f_args
    gt_audio_dir, groundtruth_tsv, use_fpn, learned_post = get_variables(f_args)

    ### edit ###
    # groundtruth_tsv = '../dataset/metadata/validation/validation.tsv'
    # gt_audio_dir = '../dataset/audio/validation'
    # use_fpn = False
    if use_fpn: 
        model_list = ['MeanTeacher_with_synthetic_fpn_shift_ICT_pseudolabel_2','MeanTeacher_with_synthetic_fpn_shift_ICT_pseudolabel_225','MeanTeacher_with_synthetic_fpn_shift_ICT_pseudolabel_25','MeanTeacher_with_synthetic_fpn_shift_ICT_pseudolabel_275','MeanTeacher_with_synthetic_fpn_shift_ICT_pseudolabel_3']
    else:
        model_list = ['MeanTeacher_with_synthetic_shift_ICT_pseudolabel_15','MeanTeacher_with_synthetic_shift_ICT_pseudolabel_2','MeanTeacher_with_synthetic_shift_ICT_pseudolabel_25']
    # save_predictions = None
    # learned_post = True
    ############
    
    dataset = DESED(base_feature_dir=osp.join(cfg.workspace, "dataset", "features"), compute_log=False)
    gt_df_feat = dataset.initialize_and_get_df(f_args.groundtruth_tsv, gt_audio_dir, nb_files=f_args.nb_files)
    
    list_predictions, params = get_predictions(model_list, gt_df_feat, f_args.save_predictions_path, learned_post=learned_post, use_fpn=use_fpn)

    groundtruth = pd.read_csv(f_args.groundtruth_tsv, sep="\t")
    meta_gt = None
    durations = generate_tsv_wav_durations(gt_audio_dir, meta_gt)
    compute_metrics(list_predictions, groundtruth, durations)
    # weak_metric = get_f_measure_by_class(params["model"], len(cfg.classes), params["weak_dataloader"])
    # print("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, params["many_hot_encoder"].labels)))
    # print("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))
    pdb.set_trace()
    
    # ##########
    # Optional but recommended
    # ##########
    # Compute psds scores with multiple thresholds (more accurate). n_thresholds could be increased.
    n_thresholds = 50
    # Example of 5 thresholds: 0.1, 0.3, 0.5, 0.7, 0.9
    list_thresholds = np.arange(1 / (n_thresholds * 2), 1, 1 / n_thresholds)
    pred_ss_thresh, _ = get_predictions(model_list, gt_df_feat, f_args.save_predictions_path, thresholds=list_thresholds, learned_post=learned_post, use_fpn=use_fpn)
    # compute_psds_from_operating_points(pred_ss_thresh, groundtruth, durations)
    psds = compute_psds_from_operating_points(pred_ss_thresh, groundtruth, durations)
    psds_score(psds, filename_roc_curves=osp.splitext(save_predictions)[0] + "_roc.png")