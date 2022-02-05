import torch.nn.functional as F
import torch
from lightning_modules.ae import AE
from utils.visualization import viz_training, viz_testing_gif, viz_testing
from utils.utils import log_average
from pytorch_msssim import ms_ssim


class AE_MSSSIM(AE):

    def __init__(self, latent_size, rho):
        super().__init__(latent_size)

        # lists to average losses over epochs
        self.training_losses = {"l1": [], "ms_ssim": []}
        self.validation_losses = {"l1": [], "ms_ssim": []}

        # balance between l1 and ms-ssim
        self.rho = rho


    ### TRAIN, VAL, TEST STEPS ###
    def training_step(self, batch, batch_idx):
        # get reconstructions, codes, residuals
        originals = batch["image"]
        reconstructions, codes = self(originals)
        residual = F.l1_loss(reconstructions, originals, reduction='none')

        # average to get loss
        l1 = torch.mean(residual)
        ms_ssim_loss = 1 - ms_ssim(originals, reconstructions, data_range=1, size_average=True, win_size=7)
        loss = self.rho * l1 + (1 - self.rho) * ms_ssim_loss

        # update GSVDD
        if batch_idx % int(len(self.trainer.train_dataloader) / 2) == 0 and batch_idx != 0:
            self.sigma = self.init_sigma()
            self.c = self.init_c()
        
        # visualization
        with torch.no_grad():
            if batch_idx % 30 == 0 and self.global_rank == 0 and self.current_epoch % 10 == 0:
                viz_training(originals, reconstructions, self.current_epoch, batch["number"], self.logger.experiment)
            self.training_losses["l1"].append(l1)
            self.training_losses["ms_ssim"].append(ms_ssim_loss)
        return loss


    def validation_step(self, batch, batch_idx):
        # get reconstructions, codes, residuals
        originals = batch["image"]
        reconstructions, _ = self(originals)
        residual = F.l1_loss(reconstructions, originals, reduction='none')

        # average to get loss
        l1 = torch.mean(residual)
        ms_ssim_loss = 1 - ms_ssim(originals, reconstructions, data_range=1, size_average=True, win_size=7)
        loss = self.rho * l1 + (1 - self.rho) * ms_ssim_loss
        self.log('val_loss', loss)  # for the scheduler       
        self.validation_losses["l1"].append(l1)
        self.validation_losses["ms_ssim"].append(ms_ssim_loss)


    def test_step(self, batch, batch_idx, dataset_idx):
        # get reconstructions, codes, residuals
        originals = batch["image"]
        labels = batch["label"]
        reconstructions, codes = self(originals)
        for visual_index in range(codes.shape[0]):
            residual = F.l1_loss(reconstructions[visual_index], originals[visual_index], reduction='none')
            if self.mean_map is not None:
                residual = torch.relu(residual - self.mean_map)        
            
            # calculate image score and feature score (GSVDD)
            l1 = torch.mean(residual, dim=[1, 2, 3])
            ms_ssim_loss = 1 - ms_ssim(originals[visual_index], reconstructions[visual_index], data_range=1, size_average=True, win_size=7)
            rec_score = self.rho * l1 + (1 - self.rho) * ms_ssim_loss
            # feat_score = torch.mean((codes - self.c) ** 2, dim=1)  # DSVDD
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

    ### LOGGING ###
    def on_validation_epoch_end(self):
        # skip on sanity check
        if not self.trainer.sanity_checking:    
            # L1
            log_average(self.training_losses, self.validation_losses, "l1", self.logger.experiment, self.current_epoch)
            # MS-SSIM
            log_average(self.training_losses, self.validation_losses, "ms_ssim", self.logger.experiment, self.current_epoch)
            
            # reset the losses buffers
            self.training_losses = {"l1": [], "ms_ssim": []}
            self.validation_losses = {"l1": [], "ms_ssim": []}
    ### END OF LOGGING ###