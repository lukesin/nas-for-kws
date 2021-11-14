# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
import os
import os.path
import random
import numpy as np
import importlib

from src.search.data_providers.base_provider import *

from definitions import DATA_PATH

torchaudio_enabled = True if importlib.util.find_spec('torchaudio') else False
# torchaudio原生支持wav和mp3格式的声音文件, 且相比于librosa, 还添加了对GPU的支持
# 有关对比可以参考 https://github.com/KinWaiCheuk/nnAudio
# nnAudio是一个全新的跨平台GPU音频处理工具: https://kinwaicheuk.github.io/nnAudio/index.html
if torchaudio_enabled:
    import torchaudio
else:
    import librosa


class SpeechCommandsDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=100, test_batch_size=100, valid_size=None,
                 n_worker=32, resize_scale=None, distort_color=None, n_mfcc=10):
        self._save_path = save_path
        self.n_mfcc = n_mfcc

        train_dataset = SpeechCommandsFolder(os.path.join(self.save_path, 'training'), augment=True, n_mfcc=self.n_mfcc)
        print(train_dataset)
        validation_dataset = SpeechCommandsFolder(os.path.join(self.save_path, 'validation'), augment=False,
                                                  n_mfcc=self.n_mfcc)
        test_dataset = SpeechCommandsFolder(os.path.join(self.save_path, 'testing'), augment=False, n_mfcc=self.n_mfcc)

        self.train = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size,
            num_workers=n_worker, pin_memory=True, shuffle=True
        )
        self.valid = torch.utils.data.DataLoader(
            validation_dataset, batch_size=train_batch_size,
            num_workers=n_worker, pin_memory=True, shuffle=True
        )
        self.test = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size,
            num_workers=n_worker, pin_memory=True,
        )

    @staticmethod
    def name():
        return 'speech_commands'

    # TODO change width
    @property
    def data_shape(self):
        return 1, self.n_mfcc, 51  # C, H, W

    # TODO add silence
    @property
    def n_classes(self):
        return 12

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join(DATA_PATH, 'speech_commands_v0.01')
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download speech commands')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(dir, class_to_idx, extensions):
    """
    数据集maker
    Args:
        dir:
        class_to_idx:
        extensions:

    Returns:

    """
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if has_file_allowed_extension(path, extensions):
                    item = (path, class_to_idx[target])
                    images.append(item)

    targets = [i[1] for i in images]
    unknown_indices = [i for i in range(len(targets)) if class_to_idx['unknown'] == targets[i]]
    silence_indices = [i for i in range(len(targets)) if class_to_idx['silence'] == targets[i]]

    all_indices = set(range(len(targets)))
    n_keywords = len(all_indices - set(unknown_indices) - set(silence_indices))
    n_unknowns = n_keywords // 10
    n_silence = n_keywords // 10

    unused_indices = random.sample(unknown_indices, len(unknown_indices) - n_unknowns)
    for i in sorted(unused_indices, reverse=True):
        del images[i]

    for i in range(n_silence - len(silence_indices)):
        silence_index = random.choice(silence_indices)
        images.append(images[silence_index])

    return images


def load_bg_data(dir):
    """

    Args:
        dir:

    Returns:

    """
    background_data = []
    background_folder = os.path.join(dir, '_background_noise_')
    for root, _, fnames in os.walk(background_folder):
        for fname in fnames:
            path = os.path.join(root, fname)
            if torchaudio_enabled:
                waveform, sample_rate = torchaudio.load(path)
                background_data.append(waveform)
            else:
                waveform, sample_rate = librosa.load(path)
                waveform = np.reshape(waveform, (1, waveform.shape[0]))
                background_data.append(torch.from_numpy(waveform))

    return background_data


class SpeechCommandsFolder(torch.utils.data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        一个以如下方式排列样本的通用数据加载器：

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): 根文件夹目录.
        loader (callable): 用于加载给定路径中的样本的函数.
        extensions (tuple[string]): 允许的扩展名 (是一个tuple). ；both 扩展名 and is_valid_file 是不被允许的.
        transform (callable, optional): 转化函数，这里认为是对输入数据做偏移用的(偏移+噪音之类的)；E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): 这个函数作者没写，不知道是为啥，猜一波上边的 transform() 已经做完了需求
        is_valid_file (callable, optional): 这个也没写，看起来是验证文件有效性的函数. 检查文件是否有效(文件是否损坏)的函数，both 扩展名 and is_valid_file 是不被允许的.

     Attributes:
        classes (list): 目标分类 (list).
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset  #Todo ?
    """

    def __init__(self, root, augment=False, n_mfcc=10):
        self.root = root
        self.augment = augment  # 扩张
        self.extensions = ('.wav',)  # 这里是允许的拓展名，torchaudio还支持mp3格式的音频文件
        self.sample_rate = 16000  # 采样率
        self.n_mfcc = n_mfcc
        self.classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        samples = make_dataset(self.root, self.class_to_idx, self.extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                self.extensions)))

        self.samples = samples
        targets = [s[1] for s in samples]
        self.targets = targets
        self.background_data = load_bg_data(root)  # Todo:

        self.unknown_indices = [i for i in range(len(targets)) if self.class_to_idx['unknown'] == targets[i]]
        self.silence_indices = [i for i in range(len(targets)) if self.class_to_idx['silence'] == targets[i]]
        all_indices = set(range(len(targets)))
        self.keyword_indices = list(all_indices - set(self.unknown_indices) - set(self.silence_indices))
        self.keyword_indices.sort()

        self.n_samples = len(all_indices)

    def loader(self, path):
        if torchaudio_enabled:
            waveform, sample_rate = torchaudio.load(path)
            n_samples = waveform.shape[1]
        else:
            waveform, sample_rate = librosa.load(path, sr=self.sample_rate)
            n_samples = waveform.shape[0]
            waveform = torch.from_numpy(np.reshape(waveform, (1, n_samples)))

        assert self.sample_rate == sample_rate

        if n_samples == sample_rate:
            return waveform
        elif n_samples < sample_rate:
            padded_waveform = torch.zeros([1, sample_rate])
            padded_waveform[0, 0:n_samples] = waveform[0]
            return padded_waveform
        elif n_samples > sample_rate:
            raise (RuntimeError("File {} has more than {} samples.".format(path, sample_rate)))

    def extract_features(self, sample):
        """
        提取特征?
        Args:
            sample:

        Returns:

        """
        if torchaudio_enabled:
            melkwargs = {
                'win_length': 640,
                'hop_length': 320,
                'n_fft': 640,
            }
            mfcc = torchaudio.transforms.MFCC(self.sample_rate, self.n_mfcc, melkwargs=melkwargs)(sample)
            return mfcc
        else:
            mfcc = librosa.feature.mfcc(sample.numpy()[0], sr=self.sample_rate, n_mfcc=self.n_mfcc, hop_length=320,
                                        n_fft=640)
            mfcc = torch.from_numpy(mfcc.astype('float32'))
            return torch.reshape(mfcc, (1, mfcc.shape[0], mfcc.shape[1]))

    def transform(self, sample):
        """
        转化?
        Args:
            sample:

        Returns:

        """
        time_shift_ms = 100
        background_frequency = 0.8
        background_volume_range = 0.1

        time_shift_samples = np.random.randint(-time_shift_ms, time_shift_ms) * self.sample_rate // 1000
        shifted_sample = torch.zeros_like(sample)
        sample_len = sample.shape[1]
        if time_shift_samples > 0:
            shifted_sample[:, time_shift_samples:] = sample[:, :sample_len - time_shift_samples]
        elif time_shift_samples < 0:
            time_shift_samples = abs(time_shift_samples)
            shifted_sample[:, :sample_len - time_shift_samples] = sample[:, time_shift_samples:]
        else:
            shifted_sample = sample

        if self.background_data:
            background_sample = random.choice(self.background_data)
            bg_sample_len = background_sample.shape[1]
            background_offset = np.random.randint(0, bg_sample_len - sample_len)
            background_clipped = background_sample[:, background_offset:(background_offset + sample_len)]
            if np.random.uniform(0, 1) < background_frequency:  # Todo:加噪声
                background_volume = np.random.uniform(0, background_volume_range)
            else:
                background_volume = 0

            transformed_sample = shifted_sample * (1 - background_volume) + background_clipped * background_volume
        else:
            transformed_sample = shifted_sample

        return transformed_sample

    def __getitem__(self, index):
        """
        如果在类中定义了__getitem__()方法，那么他的实例对象(假设为P)就可以这样P[key]取值.
        当实例对象做P[key]运算时，就会调用类中的 __getitem__() 方法, 会返回 __getitem__ 中定义的返回值.
        通常是对值进行修饰等.

        Returns:
            P[index] = sample, target
        """

        path, target = self.samples[index]
        # print(f"path, target: {path, target}")
        sample = self.loader(path)
        # print(f"sample: {sample}")
        if self.augment:  # 如果有 angment = Ture
            sample = self.transform(sample)
        sample = self.extract_features(sample)
        # print(f"sample, target: {sample, target}")
        return sample, target

    def __len__(self):
        """
        使一个类表现得像 list

        Returns:self.n_samples, 实际上是 all_indices 的长度.
        """
        return self.n_samples
