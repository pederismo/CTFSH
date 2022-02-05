import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datasets.cq500_module import CQ500DataModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from lightning_modules.ae_msssim_acai import AE_MSSSIM_ACAI
from lightning_modules.igd import IGD
from lightning_modules.vae_msssim_acai import VAE_MSSSIM_ACAI
from lightning_modules.ae import AE
from lightning_modules.vae import VAE
from lightning_modules.ae_msssim import AE_MSSSIM
from lightning_modules.vae_msssim import VAE_MSSSIM



def cli_main():
    pl.seed_everything(42, workers=True)

    # args
    parser = ArgumentParser()
    parser.add_argument('--batch_size',         default=2, type=int)
    parser.add_argument('--epochs',             default=200, type=int)
    parser.add_argument('--architecture',       default='IGD', choices=['AE', 'AE_MSSSIM', 'AE_MSSSIM_ACAI', 'VAE', 'VAE_MSSSIM', 'VAE_MSSSIM_ACAI', 'IGD'], type=str)
    parser.add_argument('--latent_size',        default=512, choices=[256, 512, 1024], type=int)
    parser.add_argument('--spatial_size',       default=128, choices=[64, 128], type=int)
    parser.add_argument('--stripped',           default=False, type=bool)
    parser.add_argument('--gpu',                default=1, type=int)
    parser.add_argument('--dataset_root',       default='.', type=str)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args() 

    # data
    cq500_root = args.dataset_root
    datamodule = CQ500DataModule(data_dir=cq500_root, stripped=args.stripped, batch_size=args.batch_size, spatial_size=args.spatial_size)

    rho = 0.15
    lambda_fool = 0.1
    gamma = 0.2
    # build the model
    if args.architecture == "AE":
        model = AE(args.latent_size)

    elif args.architecture == "VAE":
        model = VAE(args.latent_size)
    
    elif args.architecture == 'AE_MSSSIM':
        model = AE_MSSSIM(args.latent_size, rho)
    
    elif args.architecture == 'VAE_MSSSIM':
        model = VAE_MSSSIM(args.latent_size, rho)
    
    elif args.architecture == 'AE_MSSSIM_ACAI':
        model = AE_MSSSIM_ACAI(args.latent_size, rho, lambda_fool, gamma)
    
    elif args.architecture == 'VAE_MSSSIM_ACAI':
        model = VAE_MSSSIM_ACAI(args.latent_size, rho, lambda_fool, gamma)
    
    elif args.architecture == 'IGD':
        model = IGD(args.latent_size, rho, lambda_fool, gamma)
        

    # choose gpu and logger
    gpus = [args.gpu]
    experiment_name = args.architecture if not args.stripped else args.architecture + "-stripped"
    root_log_dir = os.path.join("../../../../mnt/aquarium/michele/rebuttal-raw", experiment_name)
    train_logger = TensorBoardLogger(save_dir=root_log_dir, name="pretraining")
    
    # create checkpoint callback
    checkpoint_dir = os.path.join(root_log_dir, "checkpoints") 
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}",
        save_last=True,
        every_n_epochs=10,
    )

    # create trainer object
    trainer = pl.Trainer.from_argparse_args(args, 
                                            gpus=gpus,
                                            logger=train_logger, 
                                            fast_dev_run=False,
                                            num_sanity_val_steps=0,
                                            log_every_n_steps=20,
                                            callbacks=[checkpoint_callback],
                                            max_epochs=args.epochs)
    trainer.fit(model, datamodule)


# entry point
if __name__ == '__main__':
    cli_main()