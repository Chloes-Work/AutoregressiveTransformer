from argparse import ArgumentParser, Namespace
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.optim.lr_scheduler import StepLR

from Modules.utils.features import Mel_Spectrogram

from Modules.Loss.amsoftmax import amsoftmax
from Modules.Loss.GE2ELoss import GE2ELoss
from Modules.Loss.MSELoss import MSELoss
from Modules.Loss.NCELoss import NCELoss

from Modules.utils import score

from Modules.Transformer.VanillaTransformer import VanillaTransformerEncoder
from Modules.Transformer.Transformer import Transformer

from Modules.data.speaker_encoder import SpeakerEncoder
from Modules.data.DataModul import SPKDataModul

from Modules.Scheduler.Noam import NoamScheduler


class Task(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        batch_size: int = 32,
        num_workers: int = 8,
        max_epochs: int = 1000,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mel_trans = Mel_Spectrogram()
        self.trials = np.loadtxt(self.hparams.trial_path, str)
        self.speaker_encoder = SpeakerEncoder(
            self.hparams.train_csv_path, self.hparams.valid_csv_path, self.hparams.test_csv_path)

        self.model = VanillaTransformerEncoder(output_dim=self.hparams.output_dim, embed_dim=self.hparams.embedding_dim,
                                               n_mels=self.hparams.n_mels, norm=nn.LayerNorm(self.hparams.embedding_dim, eps=1e-12))

        ## AM Soft MAX LOSS
        self.loss_fun = amsoftmax(embedding_dim=self.hparams.embedding_dim, num_classes=self.hparams.num_classes)

        # TODO fix RuntimeError: mat1 and mat2 shapes cannot be multiplied (6240x256 and 80x256)
        #self.loss_fun = GE2ELoss(embedding_dim=self.hparams.embedding_dim, num_classes=self.hparams.num_classes, device=torch.device("cuda"))

        #ODO fix  RuntimeError: mat1 and mat2 shapes cannot be multiplied (6240x256 and 80x256)
        #self.loss_fun = MSELoss()


    def forward(self, x):
        feature = self.mel_trans(x)
        embedding = self.model(feature)
        return embedding

    def training_step(self, batch, batch_idx):
        loss = self.getLosses(batch, batch_idx)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        lr = self.optimizers().param_groups[0]['lr']
        # self.log('learning_rate', lr, on_step=True, on_epoch=False, logger=True)
        if batch_idx % 100 == 0:
            print("Batch:", batch_idx, "Learning rate:", lr)

    def getEmbedding(self, batch, batch_idx):
        waveform, label, spk_id_encoded, path = batch
        feature = self.mel_trans(waveform)

        embedding = self.model(feature)
        return embedding, 0

    def getLosses(self, batch, batch_idx, train_type="train"):
        waveform, label, spk_id_encoded, path = batch
        embedding, classification = self.getEmbedding(batch, batch_idx)

        loss, acc = self.loss_fun(embedding, spk_id_encoded)
        self.log(train_type+'_loss', loss, prog_bar=True,
                 sync_dist=True, batch_size=self.hparams.batch_size)
        self.log('acc', acc, prog_bar=True,  sync_dist=True,
                 batch_size=self.hparams.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.getLosses(batch, batch_idx, train_type="Valid")
            return loss

    def on_test_epoch_start(self):
        self.index_mapping = {}
        self.eval_vectors = []
        self.trial_label = []

    def test_step(self, batch, batch_idx):
        # tiral used to compare similarity between two speakers
        trial_label, batch1, batch2 = batch
        with torch.no_grad():
            embeddings1, classification1 = self.getEmbedding(batch1, batch_idx)
            embeddings2, classification2 = self.getEmbedding(batch2, batch_idx)
            # loss = self.getLosses(batch, batch_idx)
        embeddings1 = embeddings1.detach().cpu().numpy()[0]
        embeddings2 = embeddings2.detach().cpu().numpy()[0]
        self.eval_vectors.append((embeddings1, embeddings2))
        self.index_mapping[batch_idx] = batch_idx
        self.trial_label.append(trial_label.item())


    def on_test_epoch_end(self):
        ## remove for actual run
        num_gpus = torch.cuda.device_count()

        index_mapping = {}
        if num_gpus > 1:
            eval_vectors = [None for _ in range(num_gpus)]
            dist.all_gather_object(eval_vectors, self.eval_vectors)

            table = [None for _ in range(num_gpus)]
            dist.all_gather_object(table, self.index_mapping)
            for i in table:
                index_mapping.update(i)
        else:
            eval_vectors = np.vstack(self.eval_vectors)
            index_mapping = self.index_mapping

        # https://www.isca-speech.org/archive_v0/Interspeech_2017/pdfs/0803.PDF
        # mean centering
        eval_vectors = self.eval_vectors - np.mean(self.eval_vectors, axis=0)
        scores = score.cosine_score(eval_vectors)
        EER, threshold = score.compute_eer(self.trial_label, scores)

        print("\ncosine EER: {:.2f}% with threshold {:.2f}".format(
            EER*100, threshold))
        self.log("cosine_eer", EER*100, sync_dist=True)

        minDCF, threshold = score.compute_minDCF(
            self.trial_label, scores, p_target=0.01)
        print(
            "cosine minDCF(10-2): {:.2f} with threshold {:.2f}".format(minDCF, threshold))
        self.log("cosine_minDCF(10-2)", minDCF, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = None
        if self.hparams.scheduler == 'stepLR':
            lr_scheduler = StepLR(
                optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'frequency': 1}]
        elif self.hparams.scheduler == 'noam':
            lr_scheduler = NoamScheduler(
                optimizer, self.hparams.warmup_step, self.hparams.learning_rate)
            return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}]
        if lr_scheduler is None:
            raise Exception(
                "Scheduler not set; options are stepLR or noam; set using --scheduler")

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_closure):
        # warm up learning_rate if LR_Scheduler is used
        if (self.hparams.scheduler == 'stepLR'):
            self.warmup_LR(self, optimizer)

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    @staticmethod
    def warmup_LR(self, optimizer):
        if self.trainer.global_step < self.hparams.warmup_step:
            lr_scale = min(1., float(self.trainer.global_step +
                           1) / float(self.hparams.warmup_step))
            for idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = lr_scale * self.hparams.learning_rate

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        parser.add_argument("--num_workers", default=8, type=int)
        parser.add_argument("--embedding_dim", default=252, type=int)
        parser.add_argument("--max_epochs", default=256, type=int)
        parser.add_argument("--output_dim", default=252, type=int)
        parser.add_argument("--input_dim", default=256, type=int)
        parser.add_argument("--n_mels", type=int, default=80)
        parser.add_argument("--num_classes", type=int, default=1251)
        parser.add_argument("--num_blocks", type=int, default=6)

        parser.add_argument("--input_layer", type=str, default="conv2d")
        parser.add_argument("--pos_enc_layer_type",
                            type=str, default="abs_pos")

        parser.add_argument("--second", type=int, default=3)
        parser.add_argument('--step_size', type=int, default=4)
        parser.add_argument('--gamma', type=float, default=0.5)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--warmup_step", type=float, default=2000)
        parser.add_argument("--weight_decay", type=float, default=0.0000001)
        parser.add_argument("--top_n_rows", type=int, default=None)

        parser.add_argument("--save_dir", type=str, default="./results")
        parser.add_argument("--checkpoint_path", type=str, default=None)
        parser.add_argument("--loss_name", type=str, default="amsoftmax")
        parser.add_argument("--scheduler", type=str, default="noam")

        parser.add_argument("--train_csv_path", type=str,
                            default="./train.csv")
        parser.add_argument("--valid_csv_path", type=str,
                            default="./valid.csv")
        parser.add_argument("--test_csv_path", type=str, default="./test.csv")
        parser.add_argument("--trial_path", type=str,
                            default="./veri_test2.txt")
        parser.add_argument("--score_save_path", type=str, default=None)

        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--aug', action='store_true')
        return parser


def cli_main():
    parser = Task.add_model_specific_args(ArgumentParser())
    args = parser.parse_args()

    model = Task(**args.__dict__)

    if args.checkpoint_path is not None:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")[
            "state_dict"]
        model.load_state_dict(state_dict, strict=True)
        print("load weight from {}".format(args.checkpoint_path))

    assert args.save_dir is not None
    checkpoint_callback = ModelCheckpoint(monitor='train_loss', save_top_k=100,
                                          filename="{epoch}_{train_loss:.2f}", dirpath=args.save_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # init default datamodule
    print("data augmentation {}".format(args.aug))
    dm = SPKDataModul(train_csv_path=args.train_csv_path, valid_csv_path=args.valid_csv_path,
                      test_csv_path=args.test_csv_path, speaker_encoder=model.speaker_encoder,
                      second=args.second, aug=args.aug, batch_size=args.batch_size,
                      num_workers=args.num_workers, pairs=False,
                      top_n_rows=args.top_n_rows, trial_path=args.trial_path)
    AVAIL_GPUS = torch.cuda.device_count()
    trainer = Trainer(
        max_epochs=args.max_epochs,
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator='gpu' if AVAIL_GPUS > 0 else 'cpu',
        devices=AVAIL_GPUS if AVAIL_GPUS > 0 else None,
        num_sanity_val_steps=0,
        sync_batchnorm=True if AVAIL_GPUS > 0 else False,  # Should be true when on gpu
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=args.save_dir,
        reload_dataloaders_every_n_epochs=1,
        accumulate_grad_batches=1,
        log_every_n_steps=25,
        gradient_clip_val=5
    )

    """pos_encoding = PositionalEncoding(10, 50)
    pos_encoding.plot_positional_encoding()"""
    if args.eval:
        trainer.test(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()
