import librosa
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image


def save_mel_feature(feat, name):
    plt.imshow(feat)
    plt.savefig(f'{name}.png', 'shows')
    plt.close()


class Dataset(data.Dataset):

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.data_path = cfg.data_path
        self.image_split_train = cfg.image_split_train
        self.image_split_test = cfg.image_split_test
        self.audio_split_train = cfg.audio_split_train
        self.audio_split_test = cfg.audio_split_test
        self.sample_rate = cfg.sample_rate
        self.random_flip = cfg.random_flip
        self.image_size = cfg.image_size
        self.audio_size = cfg.audio_size

        self.mean = np.array([0.485, 0.456, 0.406], dtype=float)
        self.stds = np.array([0.229, 0.224, 0.225], dtype=float)

        self.images = []
        self.labels = []
        self.audios = []

        image_split_train_files = [x.strip() for x in open(self.image_split_train).readlines()]
        audio_split_train_files = [x.strip() for x in open(self.audio_split_train).readlines()]
        self.num_samples = len(image_split_train_files)

        for line in image_split_train_files:
            image_name, label = line.split(' ')
            image_name = self.data_path + image_name
            self.images.append(image_name)
            self.labels.append(label)
        for line in audio_split_train_files:
            audio_name, _ = line.split(' ')
            audio_name = self.data_path + audio_name
            self.audios.append(audio_name)
        assert len(self.images) == len(self.labels) == len(self.audios)
        self.images = np.asarray(self.images)
        self.labels = np.asarray(self.labels)
        self.audios = np.asarray(self.audios)

        self.shuffle()

    def shuffle(self):
        shuffle_index = np.random.permutation(self.num_samples)
        shuffle_break = np.random.randint(self.num_samples)
        shuffle_shift = shuffle_index[shuffle_break:].tolist() + shuffle_index[:shuffle_break].tolist()
        shuffle_shift = np.array(shuffle_shift)
        self.shuffle_images = self.images[shuffle_shift]
        self.shuffle_audios = self.audios[shuffle_shift]

    def to_onehot(self, label):
        vector = [0 for _ in range(13)]
        vector[int(label)] = 1
        return vector

    def load_image(self, image_name):
        image = Image.open(image_name).convert('RGB')
        if self.image_size is not None:
            image = image.resize(self.image_size)
        if np.random.random() < self.random_flip and self.is_train:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = np.array(image, dtype=np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.stds
        image = image.transpose(2, 0, 1)
        return image

    def load_audio(self, audio_name):
        audio, sr = librosa.load(audio_name, sr=self.sample_rate, mono=True)
        melspec = librosa.feature.melspectrogram(audio, sr=sr, n_mels=128, hop_length=348)
        logmelspec = librosa.amplitude_to_db(melspec)
        if self.audio_size is not None:
            logmelspec = np.array(Image.fromarray(logmelspec).resize((self.audio_size[1], self.audio_size[0])))
        audio = logmelspec[..., None]
        audio /= 255.0
        audio = audio.transpose(2, 0, 1)
        return audio

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image_filename = self.images[index]
        image = self.load_image(image_filename)
        audio_filename = self.audios[index]
        audio = self.load_audio(audio_filename)
        unpair_image_filename = self.images[index]
        unpair_image = self.load_image(unpair_image_filename)
        unpair_audio_filename = self.audios[index]
        unpair_audio = self.load_audio(unpair_audio_filename)
        label = self.labels[index]
        label = np.array(self.to_onehot(label), dtype=np.float32)
        noise = np.random.uniform(-1, 1, 100).astype(np.float32)
        return image, audio, unpair_image, unpair_audio, label, noise
