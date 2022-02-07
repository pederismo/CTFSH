import torch.nn.functional as F
import torch
from lightning_modules.ae_msssim_acai import AE_MSSSIM_ACAI
from utils.visualization import viz_training, viz_testing_gif, viz_testing
from utils.utils import log_average
from pytorch_msssim import ms_ssim


class IGD(AE_MSSSIM_ACAI):

    def __init__(self, latent_size, rho, lambda_fool, gamma):
        super().__init__(latent_size, rho, lambda_fool, gamma)

        # to ensure training of encdec and discriminator 
        self.automatic_optimization = False

        # lists to average losses over epochs
        self.training_losses = {"l1": [], "ms_ssim": [], "gsvdd": []}
        self.validation_losses = {"l1": [], "ms_ssim": [], "gsvdd": []}


    ### TRAIN, VAL, TEST STEPS ###
    def training_step(self, batch, batch_idx):
        # get the optimizers
        ae_optim, disc_optim = self.optimizers()
        ae_optim.zero_grad()

        # get reconstructions and codes
        originals = batch["image"]
        reconstructions, codes = self(originals)
        
        # calculate image loss
        residual = F.l1_loss(reconstructions, originals, reduction='none')
        l1 = torch.mean(residual)
        ms_ssim_loss = 1 - ms_ssim(originals, reconstructions, data_range=1, size_average=True, win_size=7)
        image_loss = self.rho * l1 + (1 - self.rho) * ms_ssim_loss
        
        # interpolate codes
        bs = originals.shape[0]
        alpha = torch.rand(bs, 1, device=self.device).add_(-0.5).abs_()
        codes_mix = torch.lerp(
            codes, codes[range(bs - 1, -1, -1)], alpha
        )
        reconstructions_mix = self.decoder(codes_mix)
        predictions_mix = self.discriminator(reconstructions_mix)
        fooling_term = torch.sum(torch.pow(predictions_mix, 2))

        # gsvdd
        diff = (codes - self.c) ** 2
        dist = -1 * (torch.sum(diff, dim=1) / self.sigma)
        svdd_loss = torch.mean(1 - torch.exp(dist))

        # train encoder
        encoder_loss = image_loss + svdd_loss + self.lambda_fool * fooling_term
        self.manual_backward(encoder_loss)
        ae_optim.step()

        # discriminator #
        disc_optim.zero_grad()

        reconstructions_mix = self.decoder(codes_mix).detach()
        codes = self.encoder(originals)
        reconstructions = self.decoder(codes).detach()
        
        # discriminator loss
        predictions_mix = self.discriminator(reconstructions_mix)
        error_discriminator = F.mse_loss(predictions_mix, alpha.reshape(-1))

        # regularizer
        fake_interpolations = torch.lerp(reconstructions, originals, torch.tensor(self.gamma, device=self.device))
        heuristic_discriminator = torch.mean(torch.pow(self.discriminator(fake_interpolations), 2))
        
        # train the discriminator
        loss_disc = error_discriminator + heuristic_discriminator

        self.manual_backward(loss_disc)
        disc_optim.step()
        
        # update GSVDD
        if batch_idx % int(len(self.trainer.train_dataloader) / 2) == 0 and batch_idx != 0:
            print("Updated GSVDD at epoch: " + str(self.current_epoch))
            self.sigma = self.init_sigma()
            self.c = self.init_c()

        with torch.no_grad():
            # visualize and append loss
            if batch_idx < 1 and self.global_rank == 0 and self.current_epoch % 2 == 0:
                viz_training(originals, reconstructions, self.current_epoch, batch["number"], self.logger.experiment)
            
            self.log("train_loss_ae", encoder_loss, batch_size=originals.shape[0])
            self.log("train_loss_disc", loss_disc, batch_size=originals.shape[0])
            self.training_losses["l1"].append(l1.detach())
            self.training_losses["ms_ssim"].append(ms_ssim_loss.detach())
            self.training_losses['gsvdd'].append(svdd_loss.detach())
        

    def validation_step(self, batch, batch_idx):
        # get reconstructions and codes
        originals = batch["image"]
        reconstructions, codes = self(originals)
        
        # calculate image loss
        residual = F.l1_loss(reconstructions, originals, reduction='none')
        l1 = torch.mean(residual)
        ms_ssim_loss = 1 - ms_ssim(originals, reconstructions, data_range=1, size_average=True, win_size=7)
        image_loss = self.rho * l1 + (1 - self.rho) * ms_ssim_loss
        
        # interpolate codes
        bs = originals.shape[0]
        alpha = torch.rand(bs, 1, device=self.device).add_(-0.5).abs_()
        codes_mix = torch.lerp(
            codes, codes[range(bs - 1, -1, -1)], alpha
        )
        reconstructions_mix = self.decoder(codes_mix)
        predictions_mix = self.discriminator(reconstructions_mix)
        fooling_term = torch.sum(torch.pow(predictions_mix, 2))

        # gsvdd
        diff = (codes - self.c) ** 2
        dist = -1 * (torch.sum(diff, dim=1) / self.sigma)
        svdd_loss = torch.mean(1 - torch.exp(dist))

        # train encoder
        encoder_loss = image_loss + svdd_loss + self.lambda_fool * fooling_term

        reconstructions_mix = self.decoder(codes_mix).detach()
        
        # discriminator loss
        predictions_mix = self.discriminator(reconstructions_mix)
        error_discriminator = F.mse_loss(predictions_mix, alpha.reshape(-1))

        # train the discriminator
        loss_disc = error_discriminator

        # append step loss
        self.validation_losses["l1"].append(l1.detach())
        self.validation_losses["ms_ssim"].append(ms_ssim_loss.detach())
        self.validation_losses['gsvdd'].append(svdd_loss.detach())


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


    def on_validation_epoch_end(self):
        # skip on sanity check
        if not self.trainer.sanity_checking:    
            # L1
            log_average(self.training_losses, self.validation_losses, "l1", self.logger.experiment, self.current_epoch)
            # MS-SSIM
            log_average(self.training_losses, self.validation_losses, "ms_ssim", self.logger.experiment, self.current_epoch)
            # GSVDD
            log_average(self.training_losses, self.validation_losses, "gsvdd", self.logger.experiment, self.current_epoch)

            # reset the losses buffers
            self.training_losses = {"l1": [], "ms_ssim": [], "gsvdd": []}
            self.validation_losses = {"l1": [], "ms_ssim": [], "gsvdd": []}

            # update schedulers if needed
            ae_sched, disc_sched = self.lr_schedulers()
            if isinstance(ae_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                ae_sched.step(self.trainer.callback_metrics['train_loss_ae'])
            if isinstance(disc_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                disc_sched.step(self.trainer.callback_metrics['train_loss_disc'])