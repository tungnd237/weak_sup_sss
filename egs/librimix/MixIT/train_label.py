import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.models import DPRNNTasNet
from asteroid.data import LibriMix
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.losses import MixITLossWrapper, multisrc_neg_sisdr

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")

speaker_ids = ["19", "26", "27", "32", "39", "40", "60", "89"]
id_channel_mapping = {k: v for v, k in enumerate(speaker_ids)}

class MixITSystem(System):
    def training_step(self, batch, batch_nb):
        mixtures, oracle, ids = batch
        
        bsz = mixtures.shape[0]
        mix1 = mixtures[bsz // 2 :, :]
        mix2 = mixtures[: bsz // 2, :]
        moms = mix1 + mix2

        est_sources = self(moms)

        new_batch = list(range(mixtures.shape[0] // 2))
        for b in new_batch:
            if len(set([ids[0][b].split('-')[0], ids[1][b].split('-')[0], 
                ids[0][b + bsz // 2].split('-')[0], ids[1][b + bsz // 2].split('-')[0]])) != 4 :
                new_batch.remove(b)

        if len(new_batch) == 0:
            pass

        src_mix1_0_indexes = [id_channel_mapping[ids[0][b].split('-')[0]] for b in new_batch]
        src_mix1_1_indexes = [id_channel_mapping[ids[1][b].split('-')[0]] for b in new_batch]
        src_mix2_0_indexes = [id_channel_mapping[ids[0][b + bsz // 2].split('-')[0]] for b in new_batch]
        src_mix2_1_indexes = [id_channel_mapping[ids[1][b + bsz // 2].split('-')[0]] for b in new_batch]
        est_mix1 = torch.stack([est_sources[new_batch[i], mix_i, :] for i, mix_i in enumerate(src_mix1_0_indexes)], dim=0) \
            + torch.stack([est_sources[new_batch[i], mix_i, :] for i, mix_i in enumerate(src_mix1_1_indexes)], dim=0)
        est_mix2 = torch.stack([est_sources[new_batch[i], mix_i, :] for i, mix_i in enumerate(src_mix2_0_indexes)], dim=0) \
            + torch.stack([est_sources[new_batch[i], mix_i, :] for i, mix_i in enumerate(src_mix2_1_indexes)], dim=0)
        est_channels = torch.stack([est_mix1, est_mix2], dim=1)

        ref_mixes = torch.stack([mix1, mix2], dim=1)[new_batch, :, :]
        loss = self.loss_func(est_channels, ref_mixes).mean()

        tensorboard_logs = {"train_loss": loss.detach().clone()}
        self.log_dict(tensorboard_logs)

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):

        mixtures, oracle, ids = batch
        #[b, chunk_len]
        #[b, src, chunk_len]
        #[('89-219-0009', '60-121082-0058', ...), (...)]
        est_sources = self(mixtures)
        #[b, src, chunk_len]

        src_0_indexes = [id_channel_mapping[ids[0][b].split('-')[0]] for b in range(mixtures.shape[0])]
        src_1_indexes = [id_channel_mapping[ids[1][b].split('-')[0]] for b in range(mixtures.shape[0])]
        est_channels_0 = torch.stack([est_sources[b, src_0_indexes[b], :] for b in range(mixtures.shape[0])], dim=0)
        est_channels_1 = torch.stack([est_sources[b, src_1_indexes[b], :] for b in range(mixtures.shape[0])], dim=0)
        est_channels = torch.stack([est_channels_0, est_channels_1], dim=1)

        loss = self.loss_func(est_channels, oracle).mean()

        loss_dict = {"val_loss": loss.detach().clone()}
        self.log_dict(loss_dict)
        return loss_dict


def main(conf):

    assert (
        conf["training"]["batch_size"] % 2 == 0
    ), "Batch size must be divisible by two to run this recipe"

    train_set = LibriMix(
        csv_dir=conf["data"]["train_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        return_id=True,
    )

    val_set = LibriMix(
        csv_dir=conf["data"]["valid_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        return_id=True,
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    conf["masknet"].update({"n_src": len(speaker_ids)})

    model = DPRNNTasNet(
        **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"]
    )
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = multisrc_neg_sisdr
    system = MixITSystem(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        gradient_clip_val=conf["training"]["gradient_clipping"],
        log_every_n_steps=20,
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
