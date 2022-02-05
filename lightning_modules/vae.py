from typing import Any
import pytorch_lightning as pl
from models.components import EncoderVAE, Decoder
import torch
import numpy as np
import torch.nn as nn
from utils.visualization import viz_interpolation_gif, viz_training, viz_testing_gif, viz_testing
from torch.nn import functional as F
from utils.utils import Collector, log_average


class VAE(pl.LightningModule):

    def __init__(self, latent_size):
        super().__init__()

        # other parameters
        self.automatic_optimization = True
        self.visualize_testing = False
        self.mean_map = None
        self.latent_size = latent_size

        # lists to average losses over epochs
        self.training_losses = {"l1": [], "kld": []}
        self.validation_losses = {"l1": [], "kld": []}
        
        # architecture
        self.encoder = EncoderVAE(spatial_dims=3, 
                        in_shape=[1, 128, 128, 128],
                        out_channels=1,
                        latent_size=latent_size,
                        channels=[32, 64, 128, 256, 512],
                        strides=[2, 2, 2, 2, 2],
                        kernel_size=4,
                        dilation=2)
        self.decoder = Decoder(spatial_dims=3, 
                        in_shape=[1, 128, 128, 128],
                        out_channels=1,
                        final_size=self.encoder.final_size,
                        latent_size=latent_size, 
                        channels=[32, 64, 128, 256, 512],
                        strides=[1, 1, 1, 1, 1],
                        kernel_size=3)


    ### FUNCTIONS FOR GSVDD ###
    def on_train_start(self) -> None:
        self.c = self.init_c()
        self.c.requires_grad = False
        self.sigma = self.init_sigma()
        self.sigma.requires_grad = False

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['c'] = self.c
        checkpoint['sigma'] = self.sigma

    def on_load_checkpoint(self, checkpoint) -> None:
        self.c = checkpoint['c']
        self.sigma = checkpoint['sigma']

    def init_c(self, eps=0.1):
        # generator.c = None
        c = torch.zeros((1, self.latent_size)).to(self.device)
        self.encoder.eval()
        n_samples = 0
        with torch.no_grad():
            for volumes in self.trainer.train_dataloader:
                # get the inputs of the batch
                img = volumes['image'].to(self.device)
                mu, logvar = self.encoder(img)
                outputs = self.reparameterize(mu, logvar)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def init_sigma(self, sig_f=1):
        self.sigma = None
        tmp_sigma = torch.tensor(0.0, dtype=torch.float).to(self.device)
        self.encoder.eval()
        n_samples = 0
        with torch.no_grad():
            for volumes in self.trainer.train_dataloader:
                img = volumes['image'].to(self.device)
                mu, logvar = self.encoder(img)
                latent_z = self.reparameterize(mu, logvar)
                diff = (latent_z - self.c) ** 2
                tmp = torch.sum(diff.detach(), dim=1)
                if (tmp.mean().detach() / sig_f) < 1:
                    tmp_sigma += 1
                else:
                    tmp_sigma += tmp.mean().detach() / sig_f
                n_samples += 1
        tmp_sigma /= n_samples
        return tmp_sigma
    ### END GSVDD FUNCTIONS ###


    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)

        if self.training:  # multiply random noise with std only during training
            std = torch.randn_like(std).mul(std)

        return std.add_(mu)

    
    def forward(self, x: torch.Tensor, use_sigmoid: bool = True) -> Any:
        # get mu and logvar
        mu, logvar = self.encoder(x)

        # latent vector
        z = self.reparameterize(mu, logvar)
        
        # decode
        x_hat = self.decoder(z)
        return x_hat, z, mu, logvar


    ### TRAIN, VAL, TEST STEPS ###
    def training_step(self, batch, batch_idx):
        # get reconstructions, codes, residuals
        originals = batch["image"]
        reconstructions, _, mu, logvar = self(originals)
        residual = F.l1_loss(reconstructions, originals, reduction='none')
        
        # average to get reconstruction loss
        l1 = torch.mean(residual)

        # update GSVDD
        if batch_idx % int(len(self.trainer.train_dataloader) / 2) == 0 and batch_idx != 0:
            self.sigma = self.init_sigma()
            self.c = self.init_c()
        
        # add KL divergence
        bs, _, h, w, d = originals.shape
        num_voxels = h * w * d
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = l1 + kl_div

        with torch.no_grad():
            # visualize and append losses
            if batch_idx < 1 and self.global_rank == 0 and self.current_epoch % 2 == 0:
                viz_training(originals, reconstructions, self.current_epoch, batch["number"], self.logger.experiment)
            
            self.training_losses["l1"].append(l1)
            self.training_losses["kld"].append(kl_div)

        return loss


    def validation_step(self, batch, batch_idx):
        # get reconstructions and codes
        originals = batch["image"]
        reconstructions, _, mu, logvar = self(originals)
        residual = F.l1_loss(reconstructions, originals, reduction='none')
        
        # average to get reconstruction loss
        l1 = torch.mean(residual)

        # add KL divergence
        bs, _, h, w, d = originals.shape
        num_voxels = h * w * d
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = l1 + kl_div

        # append step loss
        self.log("val_loss", loss, batch_size=bs, logger=False)
        self.validation_losses["l1"].append(l1)
        self.validation_losses["kld"].append(kl_div)


    def test_step(self, batch, batch_idx, dataset_idx):
        # get reconstructions, codes, residuals
        originals = batch["image"]
        labels = batch["label"]
        reconstructions, codes, mu, logvar = self(originals)
        
        for visual_index in range(codes.shape[0]):
            residual = F.l1_loss(reconstructions[visual_index], originals[visual_index], reduction='none')
            if self.mean_map is not None:
                residual = torch.relu(residual - self.mean_map)

            # calculate image score and feature score (GSVDD)
            rec_score = torch.mean(residual, dim=[1, 2, 3])            
            diff = (codes[visual_index] - self.c.to(self.device)) ** 2
            dist = -1 * torch.sum(diff, dim=1) / self.sigma.to(self.device)
            feat_score = 1 - torch.exp(dist)       
            combined_score = (0.5 * rec_score + 0.5 * feat_score) 

            self.codes.append(codes[visual_index].detach())
            self.rec_preds.append(rec_score.detach())
            self.feat_preds.append(feat_score.detach())
            self.combined_preds.append(combined_score.detach())
            self.targets.append(labels[visual_index].detach())

            # visualize if necessary
            if self.visualize_testing:
                # discriminate dataset
                if dataset_idx == 0:
                    input_type = 'Healthy'
                else:
                    input_type = 'Unhealthy'
                
                viz_testing(originals[visual_index], reconstructions[visual_index], residual, input_type, batch['number'][visual_index],
                                rec_score, feat_score, self.thr_rec, self.thr_feat, labels[visual_index], self.logger.experiment)
                viz_testing_gif(residual, input_type + ' residual', batch["number"][visual_index], 
                                rec_score, feat_score, self.thr_rec, self.thr_feat, labels[visual_index], self.logger.experiment)
    ### END OF LIGHTNING STEPS ###
    

    def on_validation_epoch_end(self):
        # skip on sanity check
        if not self.trainer.sanity_checking:    
            # L1
            log_average(self.training_losses, self.validation_losses, "l1", self.logger.experiment, self.current_epoch)
            # KL divergence
            log_average(self.training_losses, self.validation_losses, "kld", self.logger.experiment, self.current_epoch)

            self.training_losses = {"l1": [], "kld": []}
            self.validation_losses = {"l1": [], "kld": []}
    

    def configure_optimizers(self):
        optim = torch.optim.Adam(list(self.encoder.parameters())
                                 + list(self.decoder.parameters()), lr=1e-4)
        # LR decay
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", verbose=True)
        
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }
