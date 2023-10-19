from sklearn import datasets
from sklearn.manifold import TSNE
import os
import tqdm
from mpl_toolkits.mplot3d import Axes3D
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import torch
import torch.nn.functional as F
import librosa
from audiomentations import TimeStretch, PitchShift, AddGaussianSNR, FrequencyMask, TimeMask, Reverse, Gain, Resample, \
    PolarityInversion, ApplyImpulseResponse

import utils
from net import STgramMFN
from dataset import Identity

augment_list_tiny_FPSV = [
    Identity(),
    TimeStretch(min_rate=0.75, max_rate=0.95, p=1.0),
    TimeStretch(min_rate=1.05, max_rate=1.33, p=1.0),
    PitchShift(min_semitones=-2.5, max_semitones=-0.5, p=1.0),
    PitchShift(min_semitones=0.5, max_semitones=2.5, p=1.0),
    Gain(min_gain_in_db=-2.5, max_gain_in_db=-0.5, p=1.0),
    Gain(min_gain_in_db=0.5, max_gain_in_db=2.5, p=1.0),
    AddGaussianSNR(min_snr_in_db=-2, max_snr_in_db=2, p=1.0),
    FrequencyMask(min_frequency_band=0.025, max_frequency_band=0.10, p=1.0),
    TimeMask(min_band_part=0.025, max_band_part=0.10, p=1.0),
]

augment_list_tiny_ToyConveyor = [
    Identity(),
    TimeStretch(min_rate=0.7, max_rate=0.9, p=1.0),
    TimeStretch(min_rate=1.11, max_rate=1.42, p=1.0),
    PitchShift(min_semitones=-3, max_semitones=-1, p=1.0),
    PitchShift(min_semitones=1, max_semitones=3, p=1.0),
    Gain(min_gain_in_db=-3, max_gain_in_db=-1, p=1.0),
    Gain(min_gain_in_db=1, max_gain_in_db=3, p=1.0),
    AddGaussianSNR(min_snr_in_db=-2, max_snr_in_db=2, p=1.0),
    FrequencyMask(min_frequency_band=0.025, max_frequency_band=0.10, p=1.0),
    TimeMask(min_band_part=0.025, max_band_part=0.10, p=1.0),
]

augment_list_tiny_ToyCar = [
    Identity(),
    TimeStretch(min_rate=0.6, max_rate=0.8, p=1.0),
    TimeStretch(min_rate=1.25, max_rate=1.67, p=1.0),
    PitchShift(min_semitones=-4, max_semitones=-2, p=1.0),
    PitchShift(min_semitones=2, max_semitones=4, p=1.0),
    Gain(min_gain_in_db=-4, max_gain_in_db=-2, p=1.0),
    Gain(min_gain_in_db=2, max_gain_in_db=4, p=1.0),
    AddGaussianSNR(min_snr_in_db=-2, max_snr_in_db=2, p=1.0),
    FrequencyMask(min_frequency_band=0.025, max_frequency_band=0.10, p=1.0),
    TimeMask(min_band_part=0.025, max_band_part=0.10, p=1.0),
    Reverse(p=1.0),
]

augment_dict = {
    'ToyCar': augment_list_tiny_ToyCar,
    'fan': augment_list_tiny_FPSV,
    'pump': augment_list_tiny_FPSV,
    'slider': augment_list_tiny_FPSV,
    'valve': augment_list_tiny_FPSV,
    'ToyConveyor': augment_list_tiny_ToyConveyor,
}

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# load yaml
params = utils.load_yaml(file_path='./config.yaml')
parser = argparse.ArgumentParser(description=params['description'])
for key, value in params.items():
    parser.add_argument(f'--{key}', default=value, type=type(value))
args = parser.parse_args()
args.wav2mel = utils.Wave2Mel(sr=args.sr, power=args.power,
                                      n_fft=args.n_fft, n_mels=args.n_mels,
                                      win_length=args.win_length, hop_length=args.hop_length)

def transform(filename, label):
    self = args
    # label, one_hot = utils.get_label('/'.join(filename.split('/')[-3:]), self.att2idx, self.file_att_2_idx)
    (x, _) = librosa.core.load(filename, sr=self.sr, mono=True)
    x = x[:self.sr * self.secs]
    # label = torch.randint(0, self.num_classes, size=(1,))[0]
    xs = []
    start = 0
    while start + self.win_secs <= self.secs:
        end = (start + self.win_secs) * self.sr
        xs.append(x[start * self.sr: end][np.newaxis, :])
        start += self.hop_secs
    xs = np.concatenate(xs, axis=0)
    x_wav = torch.from_numpy(augment_dict[args.machine][label](xs, sample_rate=self.sr).copy())
    # x_wav, label = self.PartAug(x)
    x_mel = self.wav2mel(x_wav)
    return x_wav, x_mel

def get_latent_features(version, epoch=120):
    # load model
    model_path = os.path.join(f'./runs/{version}/model', args.machine, f'{epoch}_checkpoint.pth.tar')
    net = STgramMFN(num_classes=args.num_classes, sec=args.win_secs, arcface=args.arcface,
                    m=args.m, s=args.s, sub=args.sub)
    net.load_state_dict(torch.load(model_path)['model'])
    net = net.to(args.device)

    # get machine list
    target_dir = args.taregt_dir
    features = []
    id_labels = []
    anomaly_labels = []

    normal_files = utils.get_filename_list(target_dir, pattern='normal_*', ext='wav')
    anomaly_files = utils.get_filename_list(target_dir, pattern='anomaly_*', ext='wav')

    for file in normal_files:
        for label in np.arange(len(augment_dict[args.machine])):
            id_labels.append(label)
            anomaly_labels.append(0)
            x_wav, x_mel = transform(file, label)
            x_wav, x_mel = x_wav.to(args.device), x_mel.to(args.device)
            label = torch.tensor([label]).to(args.device)
            with torch.no_grad():
                net.eval()
                _, feature = net(x_wav, x_mel, label)
                if args.arcface:
                    feature = F.normalize(feature)
                features.append(feature.cpu())
    for file in anomaly_files:
        label = 0
        id_labels.append(label)
        anomaly_labels.append(1)
        x_wav, x_mel = transform(file, label)
        x_wav, x_mel = x_wav.to(args.device), x_mel.to(args.device)
        label = torch.tensor([label]).to(args.device)
        with torch.no_grad():
            net.eval()
            _, feature = net(x_wav, x_mel, label)
            if args.arcface:
                feature = F.normalize(feature)
            features.append(feature.cpu())
    features = torch.cat(features, dim=0).numpy()
    return features, id_labels, anomaly_labels


def get_data(version, epoch=120):
    label_desc = [
        'Identity', 'TimeStretch_low', 'TimeStretch_high', 'PitchShift_low', 'PirchShift_high',
        'Gain_low', 'Gain_high', 'AddGaussianSNR', 'FrequencyMask', 'TimeMask', 'Reverse'
    ]

    data, id_labels, anomaly_labels = get_latent_features(version, epoch)
    data = data.reshape(data.shape[0], -1)
    return data, id_labels, anomaly_labels, label_desc


def plot_embedding(data, id_labels, anomaly_labels, label_desc, title, save_path, view='2D'):
    num_class = len(label_desc)
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    shapes = ['o', 'x']
    fig = plt.figure(figsize=(16,12))
    if view == '3D':
        ax = Axes3D(fig)
        ax.scatter(data[:, 0],
                   data[:, 1],
                   data[:, 2],
                   c=plt.cm.rainbow(id_labels / num_class),
                   s=30,
                   alpha=0.8)
    else:
        for i in range(data.shape[0]):
            plt.scatter(data[i, 0],
                        data[i, 1],
                        # str(label[i]),
                        color=plt.cm.rainbow(id_labels[i] / num_class),
                        s=40,
                        label=label_desc[id_labels[i]],
                        marker=shapes[anomaly_labels[i]],
                        alpha=0.8
                        # fontdict={'weight': 'bold', 'size': 9},
                     )

    i_list = list(range(num_class))
    patches = [mpatches.Patch(color=plt.cm.rainbow(i / num_class), label=f'{label_desc[i]}') for i in i_list]

    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    plt.legend(handles=patches, ncol=1, loc='upper right')
    plt.savefig(save_path, dpi=600)
    plt.axis('off')
    plt.show()
    plt.close()


if __name__ == '__main__':
    utils.setup_seed(512)

    version = '2022-11-26-12-(OneRange)Augmentation(sec=10)-Classifier-ArcFace(m=0.7,s=30,sub=1)'
    m_list = ['ToyCar', 'ToyConveyor', 'fan', 'pump', 'slider', 'valve']
    for machine_type in m_list:
        for mode in ['train', 'test']:
            target_data_dir = f'../../data/dataset/{machine_type}/{mode}'
            view = '2D'
            device = torch.device('cuda:0')
            save_dir = os.path.join('./tsne', version)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f't-SNE_{view}_{machine_type}_{mode}.png')


            args.machine = machine_type
            args.num_classes = len(augment_dict[machine_type])
            args.device = device
            args.taregt_dir = target_data_dir
            data, id_labels, anomaly_labels, label_desc = get_data(version, epoch='best')
            print(data.shape, len(id_labels), len(anomaly_labels))
            print('Computing t-SNE embedding')
            if view == '3D':
                tsne = TSNE(n_components=3, random_state=0, perplexity=30)
            else:
                tsne = TSNE(n_components=2, random_state=0, perplexity=20)
            result = tsne.fit_transform(data)
            plot_embedding(result, id_labels, anomaly_labels, label_desc, f't-SNE of {machine_type} latent features', save_path, view=view)