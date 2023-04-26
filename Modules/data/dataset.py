import collections
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.io import wavfile
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset


def readCSV(train_csv_path, valid_csv_path, test_csv_path):
    df_train = pd.read_csv(train_csv_path)
    df_valid = pd.read_csv(valid_csv_path)
    df_test = pd.read_csv(test_csv_path)
    return df_train, df_valid, df_test

infoPrinted = 0
def printInfo(name, n_speakers, n_utterances):
    global infoPrinted
    if infoPrinted < 2:
        print(name+" Dataset load {} speakers".format(n_speakers))
        print(name+" Dataset load {} utterance".format(n_utterances))
        infoPrinted += 1


def load_audio(filename, second=2.0):
    sample_rate, waveform = wavfile.read(filename)
    audio_length = waveform.shape[0]

    if second <= 0:
        return waveform.astype(np.float64).copy()

    length = np.int64(sample_rate * second)

    if audio_length <= length:
        shortage = length - audio_length
        waveform = np.pad(waveform, (0, shortage), 'wrap')
        waveform = waveform.astype(np.float64)
    else:
        #random start for segment length
        start = np.int64(random.random()*(audio_length-length))
        waveform =  waveform[start:start+length].astype(np.float64)
    return waveform.copy()

class TrainDataset(Dataset):
    def __init__(self, df, speaker_encoder, second=2.0, pairs=False, aug=False, top_n_rows=None, trial_path=None, dsName="Train", **kwargs):
        self.second = second
        self.pairs = pairs
        self.top_n_rows = top_n_rows
        #"ID","duration","wav","start","stop","spk_id"

        if top_n_rows:
            df = df.sample(top_n_rows)
        df = shuffle(df)
        self.labels = df["spk_id"].values
        self.paths = df["wav"].values
        self.ids = df["ID"].values
        self.speaker_encoder = speaker_encoder
        ##self.labels, self.paths = shuffle(self.labels, self.paths)
        self.trial_path = trial_path
        if self.trial_path:
            df_trial = pd.read_csv(self.trial_path, sep=" ", header=None)
            self.trials = df_trial[0], df_trial[1].str.removesuffix(".wav"), df_trial[2].str.removesuffix(".wav")

        self.aug = aug
        printInfo(dsName, len(set(self.labels)), len(set(self.paths)))
        #if aug:
            #self.wav_aug = WavAugment()

    def __getitem__(self, index):
        return self.getRow(index)
    
    def getRow(self, index):
        path = self.paths[index]
        waveform = load_audio(path, self.second)
        spk_id_encoded = self.speaker_encoder.get_speaker_label_encoded(self.labels[index])
        return torch.FloatTensor(waveform), self.labels[index], spk_id_encoded, path, 

    def __len__(self):
        if self.top_n_rows:
            return self.top_n_rows
        return len(self.paths)

class ValidDataset(TrainDataset):
    def __init__(self,df_valid,  speaker_encoder, second=2.0, pairs=True, aug=False, top_n_rows=None, **kwargs):
        super().__init__(df_valid, speaker_encoder, second, pairs, aug, top_n_rows, dsName="Valid", **kwargs)

class TestDataset(TrainDataset):
    def __init__(self,df_test,  speaker_encoder, trial_path, second=2.0, pairs=True, aug=False, top_n_rows=None, **kwargs):
        super().__init__(df_test, speaker_encoder, second, pairs, aug, top_n_rows, trial_path=trial_path,**kwargs)

    def __getitem__(self, index):
        trial_res, tri1, tri2 = self.trials
        index1 = np.argmax(self.ids == tri1[index])
        index2 = np.argmax(self.ids == tri2[index])
        return trial_res[index], self.getRow(index1), self.getRow(index2)

    def __len__(self):
        if self.top_n_rows:
            return self.top_n_rows
        return len(self.trials[0])