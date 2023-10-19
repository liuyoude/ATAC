import re
import torch
from torch.utils.data import Dataset
from audiomentations import TimeStretch, PitchShift, AddGaussianSNR, FrequencyMask, TimeMask, Reverse, Gain, Resample, \
    PolarityInversion, ApplyImpulseResponse
import torchaudio
import librosa
import numpy as np

import utils


class Identity:
    def __call__(self, wav_tensor, sample_rate=None):
        return wav_tensor

augment_list_tiny_PS = [
    Identity(),
    TimeStretch(min_rate=0.75, max_rate=0.95, p=1.0),
    TimeStretch(min_rate=1.05, max_rate=1.33, p=1.0),
    PitchShift(min_semitones=-2.5, max_semitones=-0.5, p=1.0),
    PitchShift(min_semitones=0.5, max_semitones=2.5, p=1.0),
    Gain(min_gain_in_db=-2.5, max_gain_in_db=-0.5, p=1.0),
    Gain(min_gain_in_db=0.5, max_gain_in_db=2.5, p=1.0),
    AddGaussianSNR(min_snr_in_db=-2, max_snr_in_db=2, p=1.0),
    FrequencyMask(min_frequency_band=0, max_frequency_band=0.10, p=1.0),
    TimeMask(min_band_part=0, max_band_part=0.10, p=1.0),
]

augment_list_tiny_FV = [
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
    FrequencyMask(min_frequency_band=0, max_frequency_band=0.10, p=1.0),
    TimeMask(min_band_part=0, max_band_part=0.10, p=1.0),
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
    FrequencyMask(min_frequency_band=0, max_frequency_band=0.10, p=1.0),
    TimeMask(min_band_part=0, max_band_part=0.10, p=1.0),
    Reverse(p=1.0),
]

augment_dict = {
    'ToyCar': augment_list_tiny_ToyCar,
    'fan': augment_list_tiny_FV,
    'pump': augment_list_tiny_PS,
    'slider': augment_list_tiny_PS,
    'valve': augment_list_tiny_FV,
    'ToyConveyor': augment_list_tiny_ToyConveyor,
}


class ASDDataset(Dataset):
    def __init__(self, dirs: list, args):
        self.filename_list = []
        self.sr = args.sr
        self.secs = args.secs
        self.win_secs = args.win_secs
        self.hop_secs = args.hop_secs
        self.nums = int((self.secs - self.win_secs) / self.hop_secs) + 1
        self.num_classes = args.num_classes
        for dir in dirs:
            self.filename_list.extend(utils.get_filename_list(dir))
        self.wav2mel = utils.Wave2Mel(sr=args.sr, power=args.power,
                                      n_fft=args.n_fft, n_mels=args.n_mels,
                                      win_length=args.win_length, hop_length=args.hop_length)
        self.augment = augment_dict[args.machine]
        # self.PartAug = PartAugmentation(sr=args.sr, augs_list=self.augment, secs=args.aug_secs)
        # self.random_T = RandomTransformation(sr=args.sr, num_trans=self.num_classes - 1,
        #                                      secs=self.secs, seed=args.random_t_seed)


    def __getitem__(self, item):
        filename = self.filename_list[item]
        return self.transform(filename)

    def transform(self, filename):
        # label, one_hot = utils.get_label('/'.join(filename.split('/')[-3:]), self.att2idx, self.file_att_2_idx)
        (x, _) = librosa.core.load(filename, sr=self.sr, mono=True)
        x = x[:self.sr * self.secs]
        label = torch.randint(0, self.num_classes, size=(1,))[0]
        xs = []
        start = 0
        while start + self.win_secs <= self.secs:
            end = (start + self.win_secs) * self.sr
            xs.append(x[start * self.sr: end][np.newaxis, :])
            start += self.hop_secs
        xs = np.concatenate(xs, axis=0)
        x_wav = torch.from_numpy(self.augment[label](xs, sample_rate=self.sr).copy())
        # x_wav, label = self.PartAug(x)
        # x_wav = self.random_T(xs, label).float()
        x_mel = self.wav2mel(x_wav)
        label = torch.tensor([label] * self.nums)
        return x_wav, x_mel, label

    def __len__(self):
        return len(self.filename_list)


# class PartAugmentation:
#     def __init__(self, sr, augs_list, secs=1.0):
#         self.sr = sr
#         self.augs_list = augs_list
#         self.secs = secs
#         self.num_classes = len(augs_list)
#         self.aug_len = int(secs * sr)
#
#     def __call__(self, wav_tensor):
#         wav_tensor = torch.from_numpy(wav_tensor)
#         label = torch.randint(0, self.num_classes, size=(1,))[0]
#         aug_wav = self.part_aug(wav_tensor, label)
#         return aug_wav, label
#
#     def part_aug(self, wav_tensor, label):
#         length = wav_tensor.shape[0]
#         if self.aug_len >= length:
#             aug_wav = self.augs_list[label](wav_tensor.numpy(), sample_rate=self.sr).copy()
#             return aug_wav, label
#         start = torch.randint(0, length - self.aug_len, size=(1,))[0]
#         end = start + self.aug_len
#         # print(start, end)
#         part_aug_wav = wav_tensor[start: end]
#         part_aug_wav = torch.from_numpy(self.augs_list[label](part_aug_wav.numpy(), sample_rate=self.sr).copy())
#         aug_wav = part_aug_wav
#         if start > 0:
#             left_wav = wav_tensor[:start]
#             aug_wav = torch.cat((left_wav, aug_wav), dim=0)
#         if end < length:
#             right_wav = wav_tensor[end:]
#             aug_wav = torch.cat((aug_wav, right_wav), dim=0)
#         # print(aug_wav.shape)
#         return aug_wav

# class RandomTransformation:
#     def __init__(self, sr, num_trans, secs=10.0, seed=999):
#         self.samples = sr * secs
#         self.num_classes = num_trans + 1
#         np.random.seed(seed)
#         self.w_random_list = []
#         self.b_random_list = []
#         for _ in range(self.num_classes):
#             self.w_random_list.append(np.random.randn(self.samples))
#             self.b_random_list.append(np.random.randn(self.samples))
#     def __call__(self, wav_tensor, label):
#         # wav_tensor = torch.from_numpy(wav_tensor)
#         # label = torch.randint(0, self.num_classes, size=(1,))[0]
#         aug_wav = wav_tensor if label == 0 else (wav_tensor * self.w_random_list[label] + self.b_random_list[label])
#         return torch.from_numpy(aug_wav)


if __name__ == '__main__':
    import numpy as np

    a = np.random.random((16000*2,))
    xs = [a[np.newaxis, :], a[np.newaxis, :]]
    xs = np.concatenate(xs, axis=0)
    print(xs.shape)
    # x_wavs = augment_list[2](xs, sample_rate=16000).copy()
    # print(x_wavs.shape)
    label = torch.tensor([3] * 5)
    print(label)
    #
    # part_Aug = PartAugmentation(sr=16000, augs_list=normal_augs, secs=5.0)
    # b = part_Aug(a)
