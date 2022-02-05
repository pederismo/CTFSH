from lightning_modules.vae import VAE
import torch.nn.functional as F
import torch
from utils.visualization import viz_training, viz_testing_gif, viz_testing
from pytorch_msssim import ms_ssim
from utils.utils import log_average


class VAE_MSSSIM(VAE):

    def __init__(self, latent_size, rho):
        super().__init__(latent_size)

        # lists to average losses over epochs
        self.training_losses = {"l1": [], "ms_ssim": [], "kld":[]}
        self.validation_losses = {"l1": [], "ms_ssim": [], "kld":[]}

        # balance between l1 and ms-ssim
        self.rho = rho

    ### TRAIN, VAL, TEST STEPS ###
    def training_step(self, batch, batch_idx):
        # get reconstructions, codes, residuals
        originals = batch["image"]
        reconstructions, _, mu, logvar = self(originals)
        residual = F.l1_loss(reconstructions, originals, reduction='none')
        
        # average to get reconstruction loss
        l1 = torch.mean(residual)
        ms_ssim_loss = 1 - ms_ssim(originals, reconstructions, data_range=1, size_average=True, win_size=7)
        loss = self.rho * l1 + (1 - self.rho) * ms_ssim_loss

        # update GSVDD
        if batch_idx % int(len(self.trainer.train_dataloader) / 2) == 0 and batch_idx != 0:
            self.sigma = self.init_sigma()
            self.c = self.init_c()
        
        # add KL divergence
        bs, _, h, w, d = originals.shape
        num_voxels = h * w * d
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss + kl_div

        with torch.no_grad():
            # visualize and append losses
            if batch_idx < 1 and self.global_rank == 0 and self.current_epoch % 2 == 0:
                viz_training(originals, reconstructions, self.current_epoch, batch["number"], self.logger.experiment)
            
            self.training_losses["l1"].append(l1)
            self.training_losses["kld"].append(kl_div)
            self.training_losses["ms_ssim"].append(ms_ssim_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # get reconstructions and codes
        originals = batch["image"]
        reconstructions, _, mu, logvar = self(originals)
        residual = F.l1_loss(reconstructions, originals, reduction='none')
        
        # average to get reconstruction loss
        l1 = torch.mean(residual)
        ms_ssim_loss = 1 - ms_ssim(originals, reconstructions, data_range=1, size_average=True, win_size=7)
        loss = self.rho * l1 + (1 - self.rho) * ms_ssim_loss
        
        # add KL divergence
        bs, _, h, w, d = originals.shape
        num_voxels = h * w * d
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss + kl_div
        
        # append step loss
        self.log("val_loss", loss, batch_size=bs)
        self.validation_losses["l1"].append(l1.detach())
        self.validation_losses["ms_ssim"].append(ms_ssim_loss.detach())
        self.validation_losses["kld"].append(kl_div.detach())

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
            l1 = torch.mean(residual, dim=[1, 2, 3])  
            ms_ssim_loss = 1 - ms_ssim(originals[visual_index], reconstructions[visual_index], data_range=1, size_average=False, win_size=7)
            rec_score = self.rho * l1 + (1 - self.rho) * ms_ssim_loss          
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
            # MS-SSIM
            log_average(self.training_losses, self.validation_losses, "ms_ssim", self.logger.experiment, self.current_epoch)

            self.training_losses = {"l1": [], "kld": [], "ms_ssim": []}
            self.validation_losses = {"l1": [], "kld": [], "ms_ssim": []}
    
