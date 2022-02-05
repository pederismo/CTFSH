import torch
import numpy as np
from numpy import trapz
import torchvision
import matplotlib.pyplot as plt
from monai.visualize.img2tensorboard import add_animated_gif
from utils.utils import get_classification_outcome

def viz_training(x, x_hat, epoch, indexes, logger):
    # three levels indices
    input_shape = x.shape
    index33 = int(input_shape[2] / 3)
    index50 = int(input_shape[2] / 2)
    index66 = int((2 * input_shape[2]) / 3)
    
    grid = torchvision.utils.make_grid([x[0, ..., index33], x_hat[0, ..., index33],
                                        x[1, ..., index33], x_hat[1, ..., index33],                       
                                        x[0, ..., index50], x_hat[0, ..., index50],
                                        x[1, ..., index50], x_hat[1, ..., index50],                       
                                        x[0, ..., index66], x_hat[0, ..., index66],
                                        x[1, ..., index66], x_hat[1, ..., index66]], nrow=4)
    recon_tag = "".join(["Input & Rec. at 3 levels/Scans:[", str(indexes[0].item()), ", ", str(indexes[1].item()), "], epoch:", str(epoch), ])
    logger.add_image(recon_tag, grid)


def viz_interpolation(x, x_hat, z_alpha_hat, alpha, epoch, indexes, logger):
    # three levels indices
    input_shape = x.shape
    index33 = int(input_shape[2] / 3)
    index50 = int(input_shape[2] / 2)
    index66 = int((2 * input_shape[2]) / 3)
    
    grid = torchvision.utils.make_grid([x[0, ..., index33], x_hat[0, ..., index33], z_alpha_hat[0, ..., index33],
                                        x[1, ..., index33], x_hat[1, ..., index33], z_alpha_hat[1, ..., index33], 
                                        x[0, ..., index50], x_hat[0, ..., index50], z_alpha_hat[0, ..., index50],
                                        x[1, ..., index50], x_hat[1, ..., index50], z_alpha_hat[1, ..., index50], 
                                        x[0, ..., index66], x_hat[0, ..., index66], z_alpha_hat[0, ..., index66],
                                        x[1, ..., index66], x_hat[1, ..., index66], z_alpha_hat[1, ..., index66]], nrow=3)
                      
    recon_tag = "".join(["Interpolation/[Input1, Input2, Rec.1, Rec.2, Interp.1, Interp.2], scans: [", 
                                                                                            str(indexes[0].item()), ",", str(indexes[1].item()), 
                                                                                            "], alphas: [", 
                                                                                            str(round(alpha[0].item(), 4)), ", ", str(round(alpha[1].item(), 4)), 
                                                                                            "], epoch:", str(epoch)])
    logger.add_image(recon_tag, grid)


# def viz_alpha_predictions(logger, epoch, collector):
#     predictions = collector.get_all()
#     tag = "".join(["Distribution of 20 alpha predictions at training"])
#     logger.add_histogram(tag, predictions, global_step=epoch)


def viz_interpolation_gif(codes, decoder, epoch, indexes, logger, device):
    alphas = torch.linspace(0., 0.5, steps=64, device=device)
    alphas = alphas.unsqueeze(0).unsqueeze(0).expand(2, 512, -1)
    codes_stacked = codes.unsqueeze(-1).expand(-1, -1, 64)
    interpolated = torch.lerp(codes_stacked, codes_stacked[range(1, -1, -1)], alphas)
    first_half = interpolated[0]  # 2 x 512 x 64
    second_half = interpolated[1]
    second_half = second_half.flip(dims=[-1])  # the other half of interpolations is reversed
    interpolated = torch.cat((first_half, second_half), dim=-1).transpose(0, 1) # 128 x 512

    # stack volumes together
    for i, code in enumerate(interpolated):
        code = code.unsqueeze(0)
        if i == 0:
            volumes_stacked = decoder(code)
        else:
            volumes_stacked = torch.cat([volumes_stacked, decoder(code)], dim=0)
    
    # three axial levels
    input_shape = (128, 128, 128)
    index33 = int(input_shape[2] / 3)
    index50 = int(input_shape[2] / 2)
    index66 = int((2 * input_shape[2]) / 3)

    # interpolation indices
    alpha_index0 = 0
    alpha_index1 = int(input_shape[2] / 5)
    alpha_index2 = int((2 * input_shape[2]) / 5)
    alpha_index3 = int((3 * input_shape[2]) / 5)
    alpha_index4 = int((4 * input_shape[2]) / 5)
    alpha_indexfinal = 127

    tensorone = torch.stack([volumes_stacked[alpha_index0, ..., index50],
                            volumes_stacked[alpha_index1, ..., index50],
                            volumes_stacked[alpha_index2, ..., index50],
                            volumes_stacked[alpha_index3, ..., index50],
                            volumes_stacked[alpha_index4, ..., index50],
                            volumes_stacked[alpha_indexfinal, ..., index50]])
    
    recon_tag = 'Interpolated GIF at half/Scans: [' + str(indexes[0].item()) + ", " + str(indexes[1].item()) + "], epoch: " + str(epoch)
    logger.add_images(recon_tag, tensorone)
    # gif_tag = "".join(["Interpolated GIF at half/Scans: [", str(indexes[0]), ", ", str(indexes[1]), "], epoch: ", str(epoch)])
    # add_animated_gif(logger, gif_tag, interpolated[..., index50, :].cpu(), max_out=128, scale_factor=255)
    # gif_tag = "".join(["Interpolated GIF at one third/Scans: [", str(indexes[0]), ", ", str(indexes[1]), "], epoch: ", str(epoch)])
    # add_animated_gif(logger, gif_tag, interpolated[..., index33, :].cpu(), max_out=128, scale_factor=255)


def viz_testing_gif(x, input_type, indexes, rec_score, feat_score, thr_rec, thr_feat, label, logger):
    # visualize input/reconstruction/residual in 3D GIF format
    rec_outcome = get_classification_outcome(rec_score, label, thr_rec)
    feat_outcome = get_classification_outcome(feat_score, label, thr_feat)
    
    gif_tag = "".join([input_type, "GIF/Scan:", str(indexes.item()), ", Rec:", rec_outcome, ", Feat:", feat_outcome])

    add_animated_gif(logger, gif_tag, x.cpu(), max_out=128, scale_factor=255)


def viz_testing(x, x_hat, residual, input_type, indexes, rec_score, feat_score, thr_rec, thr_feat, label, logger):
    # visualize input/reconstruction/residual in 2D at three axial levels
    input_shape = x.shape
    index33 = int(input_shape[2] / 3)
    index50 = int(input_shape[2] / 2)
    index66 = int((2 * input_shape[2]) / 3)
    
    rec_outcome = get_classification_outcome(rec_score, label, thr_rec)
    feat_outcome = get_classification_outcome(feat_score, label, thr_feat)

    grid = torchvision.utils.make_grid([x[..., index33], x_hat[..., index33], residual[..., index33],
                                        x[..., index50], x_hat[..., index50], residual[..., index50],
                                        x[..., index66], x_hat[..., index66], residual[..., index66]], nrow=3)  
    
    recon_tag = "".join([input_type + " in. & rec. & res./Scan:", str(indexes.item()), ", Rec.Outcome: ", rec_outcome, ", Feat.Outcome: ", feat_outcome])
    logger.add_image(recon_tag, grid)


def viz_pr_curve(curves, tag, logger):
    colors = ["darkorange", "g", "r", "c", "m", "y", "k", "orange", "b", "pink"]
    diseases = ["ATY", "FRAC", "ICH", "ISCH", "MASS", "OTHER", "TOTAL"]
    areas = []
    fig = plt.figure(figsize=(7, 8))

    # plot f-iso curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
    for i, curve in enumerate(curves):
        # get the metrics and compute auc
        precision, recall, thresholds = curve
        indices = ((recall == 0.5).nonzero())
        f_score = 2 * precision[indices] * recall[indices] / (precision[indices] + recall[indices])
        
    
        y = np.sort(precision)
        x = np.sort(recall)
        auc = trapz(y=y, x=x)
        areas.append(auc)

        lw = 2
        plt.plot(recall, precision, color=colors[i],
                lw=lw, label=diseases[i])

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title(tag)

    logger.add_figure(tag, fig)


def viz_confusion_matrix(matrix, tag, logger):
    fig = plt.figure(num=1)
    plt.matshow(matrix, fignum=1)
    plt.title(tag)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    logger.add_figure(tag, fig)


