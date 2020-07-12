# -*- coding: utf-8 -*-
import argparse
import datetime
import inspect
import os
import time
from pprint import pprint

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from data_utils.Desed import DESED
from data_utils.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
# from TestModel import _load_crnn
from evaluation_measures import get_predictions, psds_score, compute_psds_from_operating_points, compute_metrics, get_f_measure_by_class
# from models.CRNN import CRNN
from models import Resnet
import config as cfg
from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler import ScalerPerAudio, Scaler
from utilities.utils import SaveBest, to_cuda_if_available, weights_init, AverageMeterSet, EarlyStopping, \
    get_durations_df
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms
from tensorboardX import SummaryWriter
import torchvision.models as models
import pdb


def resnet_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Basic') != -1:
        pass
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Net_resnet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__() # necessary
        self.resnet = models.resnet18(pretrained=pretrained)
        # self.resnet = Resnet.resnet18(pretrained=pretrained)
        # convert last layer of resnet to 10 classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, len(cfg.classes))
        # convert convolution to single channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.resnet(x)
        out = self.sigmoid(x)
        return out

class Net_vgg(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__() # necessary
        self.vgg = models.vgg19_bn(pretrained=pretrained)
        self.fc = nn.Linear(1000, len(cfg.classes))
        # convert convolution to single channel
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 1-to-3 channels
        x = torch.cat((x,x,x), 1)
        x = self.vgg(x)
        x = self.fc(x)
        out = self.sigmoid(x)
        return out

def adjust_learning_rate(optimizer, rampup_value, rampdown_value=1):
    """ adjust the learning rate
    Args:
        optimizer: torch.Module, the optimizer to be updated
        rampup_value: float, the float value between 0 and 1 that should increases linearly
        rampdown_value: float, the float between 1 and 0 that should decrease linearly
    Returns:

    """
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    # We commented parts on betas and weight decay to match 2nd system of last year from Orange
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    # beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    # beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    # weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # param_group['betas'] = (beta1, beta2)
        # param_group['weight_decay'] = weight_decay


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        ema_params.data.mul_(alpha).add_(1 - alpha, params.data)


def train(train_loader, model, optimizer, c_epoch, ema_model=None, mask_weak=None, mask_strong=None, adjust_lr=False):
    """ One epoch of a Mean Teacher model
    Args:
        train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
            Should return a tuple: ((teacher input, student input), labels)
        model: torch.Module, model to be trained, should return a weak and strong prediction
        optimizer: torch.Module, optimizer used to train the model
        c_epoch: int, the current epoch of training
        ema_model: torch.Module, student model, should return a weak and strong prediction
        mask_weak: slice or list, mask the batch to get only the weak labeled data (used to calculate the loss)
        mask_strong: slice or list, mask the batch to get only the strong labeled data (used to calcultate the loss)
        adjust_lr: bool, Whether or not to adjust the learning rate during training (params in config)
    """
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    class_criterion = nn.BCELoss()
    consistency_criterion = nn.MSELoss()
    class_criterion, consistency_criterion = to_cuda_if_available(class_criterion, consistency_criterion)

    meters = AverageMeterSet()
    log.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    for i, ((batch_input, ema_batch_input), target) in enumerate(train_loader):
        global_step = c_epoch * len(train_loader) + i
        rampup_value = ramps.exp_rampup(global_step, cfg.n_epoch_rampup*len(train_loader))

        # if adjust_lr:
        #     adjust_learning_rate(optimizer, rampup_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])
        batch_input, ema_batch_input, target = to_cuda_if_available(batch_input, ema_batch_input, target)
        # Outputs
        weak_pred = model(batch_input)

        mask_unlabel = slice(6,18)
        loss = None
        # Weak BCE Loss
        # target_weak = target.max(-2)[0]  # Take the max in the time axis
        if mask_weak is not None:
            weak_class_loss = class_criterion(weak_pred[mask_strong], target[mask_strong])#class_criterion(torch.cat((weak_pred[mask_weak], weak_pred[mask_unlabel]), 0), torch.cat((target[mask_weak], target[mask_unlabel]), 0))
            loss = weak_class_loss

            if i == 0:
                log.debug(f"target: {target.mean(-2)} \n"
                        #   f"Target weak mask: {target_weak[mask_weak]} \n "
                        #   f"Target strong mask: {target[mask_strong].sum(-2)}\n"
                          f"weak loss: {weak_class_loss} \t rampup_value: {rampup_value}"
                          f"tensor mean: {batch_input.mean()}")
            meters.update('weak_class_loss', weak_class_loss.item())

        # Strong BCE loss
        if mask_strong is not None:
            strong_class_loss = class_criterion(weak_pred[mask_strong], target[mask_strong])
            meters.update('Strong loss', strong_class_loss.item())
            if loss is not None:
                loss += strong_class_loss
            else:
                loss = strong_class_loss

        # Teacher-student consistency cost
        if ema_model is not None:
            consistency_cost = cfg.max_consistency_cost * rampup_value
            '''
            meters.update('Consistency weight', consistency_cost)
            # Take consistency about strong predictions (all data)
            consistency_loss_strong = consistency_cost * consistency_criterion(strong_pred, strong_pred_ema)
            meters.update('Consistency strong', consistency_loss_strong.item())
            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong
            '''
            meters.update('Consistency weight', consistency_cost)
            # Take consistency about weak predictions (all data)
            # consistency_loss_weak = consistency_cost * consistency_criterion(weak_pred, weak_pred_ema)
            consistency_loss_weak = consistency_cost * consistency_criterion(weak_pred, weak_pred_ema)
            meters.update('Consistency weak', consistency_loss_weak.item())
            if loss is not None:
                loss += consistency_loss_weak
            else:
                loss = consistency_loss_weak
        # pdb.set_trace()
        niter = epoch * len(train_loader) + i
        writer.add_scalar('Loss', loss.item(), niter) 
        writer.add_scalar('Weak class loss', weak_class_loss.item(), niter)
        writer.add_scalar('Strong loss', strong_class_loss.item(), niter)
        
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start
    log.info(f"Epoch: {c_epoch}\t Time {epoch_time:.2f}\t {meters}")
    return loss


def get_dfs(desed_dataset, nb_files=None, separated_sources=False):
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    audio_weak_ss = None
    audio_unlabel_ss = None
    audio_validation_ss = None
    audio_synthetic_ss = None
    if separated_sources:
        audio_weak_ss = cfg.weak_ss
        audio_unlabel_ss = cfg.unlabel_ss
        audio_validation_ss = cfg.validation_ss
        audio_synthetic_ss = cfg.synthetic_ss

    weak_df = desed_dataset.initialize_and_get_df(cfg.weak, audio_dir_ss=audio_weak_ss, nb_files=nb_files)
    unlabel_df = desed_dataset.initialize_and_get_df(cfg.unlabel, audio_dir_ss=audio_unlabel_ss, nb_files=nb_files)
    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = desed_dataset.initialize_and_get_df(cfg.synthetic, audio_dir_ss=audio_synthetic_ss,
                                                       nb_files=nb_files, download=False)
    log.debug(f"synthetic: {synthetic_df.head()}")
    validation_df = desed_dataset.initialize_and_get_df(cfg.validation, audio_dir=cfg.audio_validation_dir,
                                                        audio_dir_ss=audio_validation_ss, nb_files=nb_files)
    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=0.8, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)
    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    log.debug(valid_synth_df.event_label.value_counts())

    # # Eval 2018 to report results
    # eval2018 = pd.read_csv(cfg.eval2018, sep="\t")
    # eval_2018_df = validation_df[validation_df.filename.isin(eval2018.filename)]
    # log.debug(f"eval2018 len: {len(eval_2018_df)}")

    data_dfs = {"weak": weak_df,
                "unlabel": unlabel_df,
                "synthetic": synthetic_df,
                "train_synthetic": train_synth_df,
                "valid_synthetic": valid_synth_df,
                "validation": validation_df,
                # "eval2018": eval_2018_df
                }

    return data_dfs


if __name__ == '__main__':
    torch.manual_seed(2020)
    np.random.seed(2020)
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    logger.info("Audio-tagging")
    logger.info(f"Starting time: {datetime.datetime.now()}")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")

    parser.add_argument("-n", '--no_synthetic', dest='no_synthetic', action='store_true', default=False,
                        help="Not using synthetic labels during training")

    parser.add_argument("-t", '--is_train', type=str, default='train',
                        help="train or evaluation")
    f_args = parser.parse_args()
    pprint(vars(f_args))

    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic

    is_train = f_args.is_train

    add_dir_model_name = "testest"

    store_dir = os.path.join("stored_data/audio_tagging", add_dir_model_name)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    if is_train:
        writer = SummaryWriter(os.path.join(store_dir, "log"))
        os.makedirs(store_dir, exist_ok=True)
        os.makedirs(saved_model_dir, exist_ok=True)
        os.makedirs(saved_pred_dir, exist_ok=True)

    add_axis_conv = 0

    pooling_time_ratio = 4  # 2 * 2

    # ##############
    # DATA
    # ##############
    dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                    compute_log=False)
    dfs = get_dfs(dataset, reduced_number_of_data)

    # Meta path for psds
    durations_synth = get_durations_df(cfg.synthetic)
    many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=cfg.max_frames // pooling_time_ratio)
    # encod_func = many_hot_encoder.encode_strong_df
    encod_func = many_hot_encoder.encode_weak

    # Normalisation per audio or on the full dataset
    if cfg.scaler_type == "dataset":
        transforms = get_transforms(cfg.max_frames, add_axis=add_axis_conv)
        weak_data = DataLoadDf(dfs["weak"], encod_func, transforms)
        unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms)
        train_synth_data = DataLoadDf(dfs["train_synthetic"], encod_func, transforms)
        scaler_args = []
        scaler = Scaler()
        # # Only on real data since that's our final goal and test data are real
        scaler.calculate_scaler(ConcatDataset([weak_data, unlabel_data, train_synth_data]))
        logger.debug(f"scaler mean: {scaler.mean_}")
    else:
        scaler_args = ["global", "min-max"]
        scaler = ScalerPerAudio(*scaler_args)

    transforms = get_transforms(cfg.max_frames, scaler, add_axis_conv,
                                noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv)

    weak_data = DataLoadDf(dfs["weak"], encod_func, transforms, in_memory=cfg.in_memory)
    unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms, in_memory=cfg.in_memory_unlab)
    train_synth_data = DataLoadDf(dfs["train_synthetic"], encod_func, transforms, in_memory=cfg.in_memory)
    valid_synth_data = DataLoadDf(dfs["valid_synthetic"], encod_func, transforms_valid,
                                  return_indexes=True, in_memory=cfg.in_memory)
    # real validation data
    validation_data = DataLoadDf(dfs["validation"], encod_func, transform=transforms_valid, return_indexes=True, in_memory=cfg.in_memory)
    logger.debug(f"len synth: {len(train_synth_data)}, len_unlab: {len(unlabel_data)}, len weak: {len(weak_data)}")
    # logger.debug(f"len synth: {len(train_synth_data)}, len weak: {len(weak_data)}")

    if not no_synthetic:
        list_dataset = [weak_data, unlabel_data, train_synth_data]
        batch_sizes = [cfg.batch_size//4, cfg.batch_size//2, cfg.batch_size//4]
        strong_mask = slice((3*cfg.batch_size)//4, cfg.batch_size)

    else:
        list_dataset = [weak_data, unlabel_data]
        batch_sizes = [cfg.batch_size // 4, 3 * cfg.batch_size // 4]
        strong_mask = None
    weak_mask = slice(batch_sizes[0])  # Assume weak data is always the first one

    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset, batch_sizes=batch_sizes)
    training_loader = DataLoader(concat_dataset, batch_sampler=sampler)
    valid_synth_loader = DataLoader(valid_synth_data, batch_size=cfg.batch_size)
    # real validation data
    validation_dataloader = DataLoader(validation_data, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # ##############
    # Model
    # ##############

    
    if is_train:
        model = Net_resnet(pretrained=True)

    else:
        model = Net_resnet(pretrained=False)
        model_path = os.path.join(saved_model_dir, 'baseline_best')
        expe_state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(expe_state["model"]["state_dict"])
    # pdb.set_trace()
    optim_kwargs = {"lr": cfg.default_learning_rate, "betas": (0.9, 0.999)}
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **optim_kwargs)
    # bce_loss = nn.BCELoss()

    state = {
        'model': {"name": add_dir_model_name,#crnn.__class__.__name__,
                  'args': '',
                'state_dict': model.state_dict()},
        'optimizer': {"name": optim.__class__.__name__,
                      'args': '',
                      "kwargs": optim_kwargs,
                      'state_dict': optim.state_dict()},
        "scaler": {
            "type": type(scaler).__name__,
            "args": scaler_args,
            "state_dict": scaler.state_dict()},
        "many_hot_encoder": many_hot_encoder.state_dict(),
        "desed": dataset.state_dict()
    }

    save_best_cb = SaveBest("sup")
    if cfg.early_stopping is not None:
        early_stopping_call = EarlyStopping(patience=cfg.early_stopping, val_comp="sup", init_patience=cfg.es_init_wait)

    # ##############
    # Train
    # ##############
    if is_train:
        results = pd.DataFrame(columns=["loss", "valid_synth_f1", "weak_metric", "global_valid"])
        for epoch in range(cfg.n_epoch):
            model.train()
            model = to_cuda_if_available(model)
            loss_value = train(training_loader, model, optim, epoch,
                            ema_model=None, mask_weak=weak_mask, mask_strong=strong_mask, adjust_lr=True)

            # Validation
            model = model.eval()
            logger.info("\n ### Valid synthetic metric ### \n")
            valid_synth = dfs["valid_synthetic"].drop("feature_filename", axis=1)
            weak_metric = get_f_measure_by_class(model, len(cfg.classes), valid_synth_loader, trained=True)
            print("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
            print("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))    
            writer.add_scalar('weak F1-score', np.mean(weak_metric), epoch)
            # Real validation data
            validation_labels_df = dfs["validation"].drop("feature_filename", axis=1)
            weak_metric_valid = get_f_measure_by_class(model, len(cfg.classes), validation_dataloader, trained=True)
            print("Valid weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric_valid * 100, many_hot_encoder.labels)))
            print("Valid weak F1-score macro averaged: {}".format(np.mean(weak_metric_valid)))          
            writer.add_scalar('Real Validation F1-score', np.mean(weak_metric_valid), epoch)

            # Update state
            state['model']['state_dict'] = model.state_dict()
            state['optimizer']['state_dict'] = optim.state_dict()
            state['epoch'] = epoch
            state['weak_metric'] = np.mean(weak_metric_valid)

            # Callbacks
            if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
                model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
                torch.save(state, model_fname)

            if cfg.save_best:
                if save_best_cb.apply(np.mean(weak_metric_valid)):
                    model_fname = os.path.join(saved_model_dir, "baseline_best")
                    torch.save(state, model_fname)
                results.loc[epoch, "global_valid"] = np.mean(weak_metric_valid)
            results.loc[epoch, "loss"] = loss_value.item()
            results.loc[epoch, "weak_metric"] = np.mean(weak_metric_valid)

            if cfg.early_stopping:
                if early_stopping_call.apply(np.mean(weak_metric_valid)):
                    logger.warn("EARLY STOPPING")
                    break
    # ##############
    # Evaluate
    # ##############
    else:
        model.eval()
        odel = to_cuda_if_available(model)
        validation_labels_df = dfs["validation"].drop("feature_filename", axis=1)
        weak_metric_valid = get_f_measure_by_class(model, len(cfg.classes), validation_dataloader, trained=True)
        print("Valid weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric_valid * 100, many_hot_encoder.labels)))
        print("Valid weak F1-score macro averaged: {}".format(np.mean(weak_metric_valid)))
        pdb.set_trace()       