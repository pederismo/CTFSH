import torch.nn.functional as F
import torch
from models.components import Discriminator
from lightning_modules.ae_msssim import AE_MSSSIM
from utils.visualization import viz_training, viz_testing_gif, viz_testing
from utils.utils import log_average
from pytorch_msssim import ms_ssim


class AE_MSSSIM_ACAI(AE_MSSSIM):

    def __init__(self, latent_size, rho, lambda_fool, gamma):
        super().__init__(latent_size, rho)

        # to ensure training of encdec and discriminator 
        self.automatic_optimization = False

        # lists to average losses over epochs
        self.training_losses = {"l1": [], "ms_ssim": [], "fool_term": [], "discr_error": [], "discr_heurist": []}
        self.validation_losses = {"l1": [], "ms_ssim": [], "fool_term": [], "discr_error": [], "discr_heurist": []}

        # discriminator
        self.discriminator = Discriminator(spatial_dims=3,
                                           in_shape=[1, 128, 128, 128],
                                           out_channels=1,
                                           latent_size=512,
                                           channels=[32, 64, 128, 256, 512],
                                           strides=[2, 2, 2, 2, 2],
                                           kernel_size=4,
                                           dilation=2)

        # to balance the fooling term and regularize discriminator
        self.lambda_fool = lambda_fool
        self.gamma = gamma


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

        # train encoder
        encoder_loss = image_loss + self.lambda_fool * fooling_term
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
            self.training_losses["fool_term"].append(fooling_term.detach())
            self.training_losses["discr_error"].append(error_discriminator.detach())
            self.training_losses["discr_heurist"].append(heuristic_discriminator.detach())

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
        
        # discriminator loss
        reconstructions_mix = self.decoder(codes_mix).detach()
        predictions_mix = self.discriminator(reconstructions_mix)
        error_discriminator = F.mse_loss(predictions_mix, alpha.reshape(-1))

        # regularizer
        fake_interpolations = torch.lerp(reconstructions, originals, torch.tensor(self.gamma, device=self.device))
        heuristic_discriminator = torch.mean(torch.pow(self.discriminator(fake_interpolations), 2))
        
        # append step loss
        self.validation_losses["l1"].append(l1.detach())
        self.validation_losses["ms_ssim"].append(ms_ssim_loss.detach())
        self.validation_losses["fool_term"].append(fooling_term.detach())
        self.validation_losses["discr_error"].append(error_discriminator.detach())
        self.validation_losses["discr_heurist"].append(heuristic_discriminator.detach())
    ### END OF LIGHTNING STEPS ###


    def on_validation_epoch_end(self):
        # skip on sanity check
        if not self.trainer.sanity_checking:    
            # L1
            log_average(self.training_losses, self.validation_losses, "l1", self.logger.experiment, self.current_epoch)
            # MS-SSIM
            log_average(self.training_losses, self.validation_losses, "ms_ssim", self.logger.experiment, self.current_epoch)
            # Fooling term
            log_average(self.training_losses, self.validation_losses, "fool_term", self.logger.experiment, self.current_epoch)
            # Error discriminator
            log_average(self.training_losses, self.validation_losses, "discr_error", self.logger.experiment, self.current_epoch)
            # Heuristic discriminator
            log_average(self.training_losses, self.validation_losses, "discr_heurist", self.logger.experiment, self.current_epoch)

            # reset the losses buffers
            self.training_losses = {"l1": [], "ms_ssim": [], "fool_term": [], "discr_error": [], "discr_heurist": []}
            self.validation_losses = {"l1": [], "ms_ssim": [], "fool_term": [], "discr_error": [], "discr_heurist": []}

            # update schedulers if needed
            ae_sched, disc_sched = self.lr_schedulers()
            if isinstance(ae_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                ae_sched.step(self.trainer.callback_metrics['train_loss_ae'])
            if isinstance(disc_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                disc_sched.step(self.trainer.callback_metrics['train_loss_disc'])

    
    def configure_optimizers(self):
        ae_optim_dict =  super().configure_optimizers()

        # optimizer & scheduler for discriminator
        disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        disc_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(disc_optim, mode='min', verbose=True)

        return (
            ae_optim_dict,
            {
                'optimizer': disc_optim,
                'lr_scheduler': {
                    'scheduler': disc_sched,
                    'monitor': 'train_loss_disc',
                }
            }
        )