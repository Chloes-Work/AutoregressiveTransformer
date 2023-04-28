from typing import Any
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from Modules.data.dataset import TrainDataset, ValidDataset, TestDataset, readCSV

class SPKDataModul(LightningDataModule):
    def __init__(
        self,
        train_csv_path,
        valid_csv_path,
        test_csv_path,
        speaker_encoder,
        second: int = 2.0,
        num_workers: int = 8,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        pairs: bool = True,
        aug: bool = False,
        semi: bool = False,
        top_n_rows: int = None,
        trial_path = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        df_train, df_valid, df_test = readCSV(train_csv_path, valid_csv_path, test_csv_path)
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.top_n_rows = top_n_rows
        self.second = second
        self.pairs = pairs
        self.aug = aug
        self.speaker_encoder = speaker_encoder
        self.trial_path = trial_path
        print("second is {:.2f}".format(second))

    def train_dataloader(self) -> DataLoader:
        train_dataset = TrainDataset(self.df_train, self.speaker_encoder, self.second, self.pairs, self.aug, self.top_n_rows)
        loader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
                drop_last=False,
                )
        return loader

    def val_dataloader(self) -> DataLoader:
        valid_dataset = ValidDataset(self.df_valid, self.speaker_encoder, self.second, self.pairs, self.aug, self.top_n_rows)
        loader = torch.utils.data.DataLoader(
                valid_dataset,
                shuffle=False,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
                drop_last=False,
                )
        return loader

    def test_dataloader(self) -> DataLoader:
        #test = np.loadtxt(self.test_csv_path, str)
        #self.test = test
        #testpath = np.unique(np.concatenate((test.T[1], test.T[2])))
        #print("number of enroll: {}".format(len(set(test.T[1]))))
        #print("number of test: {}".format(len(set(test.T[2]))))
        #print("number of evaluation: {}".format(len(testpath)))
        test_dataset = TestDataset(self.df_test, self.speaker_encoder, self.trial_path, self.second, self.pairs, self.aug, self.top_n_rows)
        loader = torch.utils.data.DataLoader(test_dataset,
                                             num_workers=8,
                                             shuffle=False, 
                                             batch_size=1)
        return loader