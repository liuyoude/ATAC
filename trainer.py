import os
import glob
import librosa
import sklearn
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
import joblib

from loss import ASDLoss
from dataset import augment_dict
import utils


class Trainer:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.net = kwargs['net']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = self.args.writer
        self.logger = self.args.logger
        self.criterion = ASDLoss(num_classes=self.args.num_classes, w_normal=self.args.w_normal).to(self.args.device)
        self.wav2mel = utils.Wave2Mel(sr=self.args.sr, power=self.args.power,
                                      n_fft=self.args.n_fft, n_mels=self.args.n_mels,
                                      win_length=self.args.win_length, hop_length=self.args.hop_length)
        self.augment = augment_dict[self.args.machine]
        # self.PartAug = PartAugmentation(sr=self.args.sr, augs_list=self.augment, secs=self.args.aug_secs)
        self.nums = int((self.args.secs - self.args.win_secs) / self.args.hop_secs) + 1
        # self.nums_augment = self.args.num_classes
        self.nums_augment = 1 # only use Identity() to valid and test
    def train(self, train_loader, valid_dir):
        # self.valid(valid_dir, save=False)
        model_dir = os.path.join(self.writer.log_dir, 'model', self.args.machine)
        os.makedirs(model_dir, exist_ok=True)
        epochs = self.args.epochs
        valid_every_epochs = self.args.valid_every_epochs
        early_stop_epochs = self.args.early_stop_epochs
        start_valid_epoch = self.args.start_valid_epoch
        num_steps = len(train_loader)
        self.sum_train_steps = 0
        self.sum_valid_steps = 0
        best_metric = 0
        no_better_epoch = 0
        for epoch in range(0, epochs + 1):
            # train
            sum_loss = 0
            self.net.train()
            train_bar = tqdm(train_loader, total=num_steps, desc=f'{self.args.machine}|Epoch-{epoch}')
            for (x_wavs, x_mels, labels) in train_bar:
                # forward
                b, n, f, t = x_mels.shape
                x_wavs, x_mels = x_wavs.reshape(b*n, -1).float().to(self.args.device), \
                                 x_mels.reshape(b*n, f, t).float().to(self.args.device)
                labels = labels.reshape(-1).long().to(self.args.device)
                # print(x_wavs.shape, x_mels.shape, labels.shape)
                out_classes, zs = self.net(x_wavs, x_mels, labels)
                loss = self.criterion(out_classes, labels)
                train_bar.set_postfix(loss=f'{loss.item():.5f}')
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # visualization
                self.writer.add_scalar(f'{self.args.machine}/train_loss', loss.item(), self.sum_train_steps)
                sum_loss += loss.item()
                self.sum_train_steps += 1
            avg_loss = sum_loss / num_steps
            if self.scheduler is not None and epoch >= self.args.start_scheduler_epoch:
                self.scheduler.step()
            self.logger.info(f'Epoch-{epoch}\tloss:{avg_loss:.5f}')
            # valid
            if (epoch - start_valid_epoch) % valid_every_epochs == 0 and epoch >= start_valid_epoch:
                metric, _ = self.valid(valid_dir, save=False)
                avg_auc, avg_pauc = metric['avg_auc'], metric['avg_pauc']
                self.writer.add_scalar(f'{self.args.machine}/auc_s', avg_auc, epoch)
                self.writer.add_scalar(f'{self.args.machine}/pauc', avg_pauc, epoch)
                if avg_auc + avg_pauc >= best_metric:
                    no_better_epoch = 0
                    best_metric = avg_auc + avg_pauc
                    best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)
                    self.logger.info(f'Best epoch now is: {epoch:4d}')
                else:
                    # early stop
                    no_better_epoch += 1
                    if no_better_epoch > early_stop_epochs > 0: break
            # save last 10 epoch state dict
            if epoch >= self.args.start_save_model_epochs:
                if (epoch - self.args.start_save_model_epochs) % self.args.save_model_interval_epochs == 0:
                    model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
                    utils.save_model_state_dict(model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)

    def valid(self, valid_dir, save=True, result_dir=None, csv_lines=[]):
        net = self.net.module if self.args.dp else self.net
        net.eval()
        metric = {}
        print('\n' + '=' * 20)
        result_dir = result_dir if result_dir else os.path.join('./results', self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        target_dir = valid_dir
        start = time.perf_counter()
        machine_type = target_dir.split('/')[-2]
        machine_id_list = utils.get_machine_id_list(target_dir)
        # print(machine_id_list)
        csv_lines.append([machine_type])
        csv_lines.append(['ID', 'AUC', 'pAUC'])
        performance = []
        for id_str in machine_id_list:
            csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
            test_files, y_true = utils.get_valid_file_list(target_dir, id_str)
            y_pred = [0. for _ in test_files]
            anomaly_score_list = []
            for file_idx, file_path in enumerate(test_files):
                x_wavs, x_mels, labels = self.transform(file_path)
                with torch.no_grad():
                    out_classes, _ = net(x_wavs, x_mels, labels)
                nlls = - torch.log_softmax(out_classes, dim=1).cpu().numpy().reshape(-1, self.nums,
                                                              self.args.num_classes)
                # mean in each segment
                nlls = nlls.mean(axis=1)
                # anomaly socre
                y_pred[file_idx] = nlls[0][0]

                # mat_eye = np.eye(self.args.num_classes, dtype=bool)
                # # y_pred[file_idx] = np.mean(aug_anomaly_score_list)
                # if self.args.dirichlet_score and os.path.exists(self.args.dirichlet_path):
                #     alphas = np.array(joblib.load(self.args.dirichlet_path))
                #     y_pred[file_idx] = ((alphas - 1) * nlls).mean(axis=0).mean(axis=0)
                # else:
                #     y_pred[file_idx] = np.mean(nlls[mat_eye][:self.nums_augment])

                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
            if save: utils.save_csv(csv_path, anomaly_score_list)
            # compute auc and pAuc
            auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
            p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
            performance.append([auc, p_auc])
            csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
        # calculate averages for AUCs and pAUCs
        # print(performance)
        amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
        mean_auc, mean_p_auc = amean_performance[0], amean_performance[1]
        # print(machine_type, 'AUC_clf:', mean_auc, 'pAUC_clf:', mean_p_auc)
        time_nedded = time.perf_counter() - start
        csv_lines.append(["Average"] + list(amean_performance))
        csv_lines.append([])
        self.logger.info(f'Test {machine_type}\tcost {time_nedded:.2f} sec\tavg_auc: {mean_auc:.3f}\tavg_pauc: {mean_p_auc:.3f}')
        print(f'Test time: {time_nedded:.2f} sec')
        metric['avg_auc'], metric['avg_pauc'] = mean_auc, mean_p_auc
        return metric, csv_lines

    def test(self, test_dir, result_dir=None):
        net = self.net.module if self.args.dp else self.net
        net.eval()
        metric = {}
        print('\n' + '=' * 20)
        result_dir = result_dir if result_dir else os.path.join('./evaluator/teams', self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        target_dir = test_dir
        machine_type = target_dir.split('/')[-2]
        machine_id_list = utils.get_machine_id_list(target_dir)
        for id_str in machine_id_list:
            csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
            test_files = utils.get_test_file_list(target_dir, id_str)
            y_pred = [0. for _ in test_files]
            anomaly_score_list = []
            for file_idx, file_path in enumerate(test_files):
                x_wavs, x_mels, labels = self.transform(file_path)
                with torch.no_grad():
                    out_classes, _ = net(x_wavs, x_mels, labels)
                nlls = - torch.log_softmax(out_classes, dim=1).cpu().numpy()
                # mean in each segment
                nlls = nlls.mean(axis=0)

                # aug_anomaly_score_list = nlls[mat_eye]
                # y_pred[file_idx] = aug_anomaly_score_list[0]
                y_pred[file_idx] = nlls[0]
                # y_pred[file_idx] = np.mean(aug_anomaly_score_list)
                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
            utils.save_csv(csv_path, anomaly_score_list)


    def cal_dirichlet(self, train_dir):
        net = self.net.module if self.args.dp else self.net
        net.eval()
        target_dir = train_dir
        saved_path = self.args.dirichlet_path
        saved_dir, _ = os.path.split(saved_path)
        os.makedirs(saved_dir, exist_ok=True)
        file_pattern = os.path.join(target_dir, '*.wav')
        train_files = glob.glob(file_pattern)
        probs_dict = {}
        for label in range(self.args.num_classes):
            probs_dict[label] = []
        pbar = tqdm(train_files, total=len(train_files), desc="Calculate alphas of dirichlet")
        for train_file in pbar:
            x_wavs, x_mels, labels = self.transform(train_file)
            with torch.no_grad():
                out_classes, _ = net(x_wavs, x_mels, labels)
            probs = torch.softmax(out_classes, dim=1).cpu().numpy()
            for label in range(self.args.num_classes):
                probs_dict[label].append(probs[label])
        alphas = []
        for label in range(self.args.num_classes):
            observed_dirichlet = np.array(probs_dict[label])
            log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)
            alpha_sum_approx = utils.calc_approx_alpha_sum(observed_dirichlet)
            alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx
            mle_alpha_t = utils.fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
            alphas.append(mle_alpha_t)
        self.logger.info('Dirichlet alphas:')
        self.logger.info(alphas)
        with open(saved_path, 'wb') as f:
            joblib.dump(alphas, f)


    def transform(self, filename):
        # label, one_hot = utils.get_label('/'.join(filename.split('/')[-3:]), self.att2idx, self.file_att_2_idx)
        (x, _) = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[:self.args.sr * 10]  # (1, audio_length)
        xs = []
        start = 0
        while start + self.args.win_secs <= self.args.secs:
            end = (start + self.args.win_secs) * self.args.sr
            xs.append(x[start * self.args.sr: end][np.newaxis, :])
            start += self.args.hop_secs
        xs = np.concatenate(xs, axis=0)
        x_wav_augs = []
        for aug_idx in range(self.nums_augment):
            x_wav_aug = torch.from_numpy(self.augment[aug_idx](xs, sample_rate=self.args.sr).copy())
            # x_wav_aug = self.PartAug.part_aug(torch.from_numpy(x), aug_idx).unsqueeze(0)
            x_wav_augs.append(x_wav_aug)
        x_wav_augs = torch.cat(x_wav_augs, dim=0)
        x_mel_augs = self.wav2mel(x_wav_augs)
        # 1 batch input
        x_wav_augs = x_wav_augs.float().to(self.args.device)
        x_mel_augs = x_mel_augs.float().to(self.args.device)
        labels = torch.from_numpy(np.array([[idx] * self.nums for idx in range(self.nums_augment)])).reshape(-1).long().to(self.args.device)
        return x_wav_augs, x_mel_augs, labels
