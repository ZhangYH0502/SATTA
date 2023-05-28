import torch
import numpy as np
import os
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


def make_one_hot(num_labels, labels):
    label_shape = np.array(labels.shape)
    label_shape[1] = num_labels
    label_shape = tuple(label_shape)
    labels_onehot = torch.zeros(label_shape)
    labels_onehot = labels_onehot.scatter_(1, labels, 1)
    return labels_onehot


def dice_coeff(all_predictions, all_targets, smooth=1e-5):
    dice_array = np.zeros((3, 2), dtype=float)

    batch_num = all_targets.shape[0]

    for i in range(3):
        pred = all_predictions.clone() == (i+1)
        targ = all_targets.clone() == (i+1)

        pred = pred.contiguous().view(batch_num, -1).numpy()
        targ = targ.contiguous().view(batch_num, -1).numpy()

        false_idx = (np.sum(targ, axis=-1) > 0)

        pred = pred[false_idx, :]
        targ = targ[false_idx, :]

        intersection = (pred * targ).sum(-1)

        union = pred.sum(-1) + targ.sum(-1)

        dsc = (2. * intersection + smooth) / (union + smooth)

        dice_array[i, 0] = np.mean(dsc)
        dice_array[i, 1] = np.std(dsc)

    return dice_array


def compute_metrics(all_predictions, all_targets, loss, verbose=True):

    dice_array = dice_coeff(all_predictions, all_targets)

    total_mean = np.mean(dice_array[:, 0])
    total_std = np.mean(dice_array[:, 1])

    if verbose:
        print('loss: {:0.3f}'.format(loss))
        print('total_mDSC: {:0.1f}'.format(total_mean * 100))
        print('IRF_mDSC: {:0.1f}'.format(dice_array[0, 0] * 100))
        print('SRF_mDSC: {:0.1f}'.format(dice_array[1, 0] * 100))
        print('PED_mDSC: {:0.1f}'.format(dice_array[2, 0] * 100))

    metrics_dict = {}
    metrics_dict['loss'] = loss
    metrics_dict['total_mDSC'] = total_mean
    metrics_dict['total_vDSC'] = total_std
    metrics_dict['IRF_mDSC'] = dice_array[0, 0]
    metrics_dict['IRF_vDSC'] = dice_array[0, 1]
    metrics_dict['SRF_mDSC'] = dice_array[1, 0]
    metrics_dict['SRF_vDSC'] = dice_array[1, 1]
    metrics_dict['PED_mDSC'] = dice_array[2, 0]
    metrics_dict['PED_vDSC'] = dice_array[2, 1]

    return metrics_dict


def compute_metrics_salr(all_predictions, all_targets, verbose=True):

    dice_array = dice_coeff(all_predictions, all_targets)

    total_mean = np.mean(dice_array[:, 0])
    total_std = np.mean(dice_array[:, 1])

    if verbose:
        print('total_mDSC: {:0.1f}'.format(total_mean * 100))
        print('IRF_mDSC: {:0.1f}'.format(dice_array[0, 0] * 100))
        print('SRF_mDSC: {:0.1f}'.format(dice_array[1, 0] * 100))
        print('PED_mDSC: {:0.1f}'.format(dice_array[2, 0] * 100))

    metrics_dict = {}
    metrics_dict['total_mDSC'] = total_mean
    metrics_dict['total_vDSC'] = total_std
    metrics_dict['IRF_mDSC'] = dice_array[0, 0]
    metrics_dict['IRF_vDSC'] = dice_array[0, 1]
    metrics_dict['SRF_mDSC'] = dice_array[1, 0]
    metrics_dict['SRF_vDSC'] = dice_array[1, 1]
    metrics_dict['PED_mDSC'] = dice_array[2, 0]
    metrics_dict['PED_vDSC'] = dice_array[2, 1]

    return metrics_dict


class WriteLog:
    def __init__(self, model_name):
        self.model_name = model_name
        open(model_name+'/train.log', "w").close()
        open(model_name+'/valid.log', "w").close()
        open(model_name+'/test.log', "w").close()

    def log_losses(self, file_name, epoch, loss, metrics):
        log_file = open(self.model_name+'/'+file_name, "a")
        log_file.write(str(epoch) + ',  ' + str(round(loss, 4)) + ',  '
                       + str(round(metrics['total_mDSC'], 4)) + '+' + str(round(metrics['total_vDSC'], 4)) + ',  '
                       + str(round(metrics['IRF_mDSC'], 4)) + '+' + str(round(metrics['IRF_vDSC'], 4)) + ',  '
                       + str(round(metrics['SRF_mDSC'], 4)) + '+' + str(round(metrics['SRF_vDSC'], 4)) + ',  '
                       + str(round(metrics['PED_mDSC'], 4)) + '+' + str(round(metrics['PED_vDSC'], 4)) + '\n')
        log_file.close()


class SaveBestModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.best_mDSC = 0
        self.count = 0

    def evaluate(self, valid_metrics, test_metrics, epoch, model, all_preds, all_ids):

        if valid_metrics['total_mDSC'] > self.best_mDSC:
            self.best_mDSC = valid_metrics['total_mDSC']

            print('> Saving Model\n')
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'valid_mDSC': valid_metrics['total_mDSC'],
                'test_mDSC': test_metrics['total_mDSC'],
            }
            torch.save(save_dict, self.model_name + '/best_model.pt')

            print('\n')
            print('best total_mDSC:  {:0.1f}'.format(test_metrics['total_mDSC'] * 100))
            print('best IRF_mDSC:  {:0.1f}'.format(test_metrics['IRF_mDSC'] * 100))
            print('best SRF_mDSC:  {:0.1f}'.format(test_metrics['SRF_mDSC'] * 100))
            print('best PED_mDSC:  {:0.1f}'.format(test_metrics['PED_mDSC'] * 100))

            mask_save_path = self.model_name + "/" + "result_masks"
            if not os.path.exists(mask_save_path):
                os.makedirs(mask_save_path)

            all_preds = all_preds.numpy()

            for i in range(all_preds.shape[0]):
                img = all_preds[i, :, :] * 80

                img_id = all_ids[i]

                im = Image.fromarray(np.uint8(img))
                im.save(mask_save_path + "/" + img_id)

            self.count = 0

        else:
            self.count = self.count + 1

        # if self.count > 10:
        #     print("no performance improvement within 10 epoches")
        #     exit(0)


if __name__ == "__main__":

    x = np.random.rand(3, 5)
    pred = (x > 0.6).astype(int)
    targ = (x > 0.3).astype(int)

    print(pred)
    print(targ)

    intersection = (pred * targ).sum(-1)
    print(intersection)

    union = pred.sum(-1) + targ.sum(-1)
    print(union)

    dsc = (2. * intersection + 1e-5) / (union + 1e-5)
    print(dsc)




