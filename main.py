import torch
import torch.nn as nn

import argparse
import os
from tqdm import tqdm
import numpy as np
from PIL import Image

import utils
from load_data import get_data
from config_args import get_args
import loss_function as lf
import tent

from UNets import U_Net, U_Net_Uncertainty
from utils1 import entropy_uncertainly


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
# net.apply(apply_dropout)


def model_inference(args, net, dataloader):
    model1 = tent.configure_model(net)
    params, param_names = tent.collect_params(model1)
    optimizer = torch.optim.Adam(params, lr=0.00001)
    tented_net = tent.Tent(model1, optimizer)

    all_logits, all_preds, all_targs, all_ids, test_loss = run_epoch(tented_net, dataloader, None, 'Testing', False, True)
    # all_preds, all_targs, all_ids, test_loss = run_epoch_with_train(net, dataloader, 'Testing', False)

    # outputs = entropy_uncertainly(all_logits, all_ids, args.model_name)

    print('\n')
    print('output test metric:')
    test_metrics = utils.compute_metrics(all_preds, all_targs, test_loss)

    mask_save_path = args.model_name + "/" + "result_masks_tent"
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)
    all_preds = all_preds.numpy()
    for i in range(all_preds.shape[0]):
        img = all_preds[i, :, :] * 80
        img_id = all_ids[i]
        im = Image.fromarray(np.uint8(img))
        im.save(mask_save_path + "/" + img_id)


def model_train(args, net, train_loader, valid_loader, test_loader):
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)  # weight_decay=0.0004)
    step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    write_log = utils.WriteLog(args.model_name)
    save_best_model = utils.SaveBestModel(args.model_name)

    for epoch in range(1, args.epochs+1):
        print('======================== {} ========================'.format(epoch))
        for param_group in optimizer.param_groups:
            print('LR: {}'.format(param_group['lr']))

        train_loader.dataset.epoch = epoch

        ################### Train #################
        all_logits_train, all_preds_train, all_targs_train, all_ids_train, train_loss = run_epoch(net, train_loader, optimizer, 'Training', True, False)
        print('\n')
        print('output train metric:')
        train_metrics = utils.compute_metrics(all_preds_train, all_targs_train, train_loss)
        write_log.log_losses('train.log', epoch, train_loss, train_metrics)

        ################### Valid #################
        all_logits_valid, all_preds_valid, all_targs_valid, all_ids_valid, valid_loss = run_epoch(net, valid_loader, None, 'Validating', False, False)
        print('\n')
        print('output valid metric:')
        valid_metrics = utils.compute_metrics(all_preds_valid, all_targs_valid, valid_loss)
        write_log.log_losses('valid.log', epoch, valid_loss, valid_metrics)

        ################### Test #################
        if test_loader is not None:
            all_preds_test, all_targs_test, all_ids_test, test_loss = run_epoch(net, test_loader, None, 'Testing', False, False)
            print('\n')
            print('output test metric:')
            test_metrics = utils.compute_metrics(all_preds_test, all_targs_test, test_loss)
            write_log.log_losses('test.log', epoch, test_loss, test_metrics)
        else:
            test_metrics = valid_metrics
            test_loss = valid_loss

        step_scheduler.step(valid_loss)

        ############## Log and Save ##############
        save_best_model.evaluate(valid_metrics, test_metrics, epoch, net, all_preds_valid, all_ids_valid)


def run_epoch(net, data, optimizer, desc, train=False, tent=False):
    if train:
        net.train()
        optimizer.zero_grad()
    else:
        if tent is False:
            net.eval()

    criterion = nn.CrossEntropyLoss()

    all_predictions = []
    all_targets = []
    all_logits = []
    all_image_ids = []
    loss_total = []

    for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):

        images = batch['images'].float()
        masks = batch['masks']
        # class_labels = batch['class_labels']

        if train:
            pred = net(images.cuda())
        else:
            if tent:
                pred = net(images.cuda())
            else:
                with torch.no_grad():
                    pred = net(images.cuda())

        loss = criterion(pred, masks.cuda())

        loss_total.append(loss.item())

        # print('backward...')
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        pred_arg = torch.argmax(pred.clone().detach(), dim=1)

        batch_num = masks.shape[0]
        for i in range(batch_num):
            all_logits.append(pred[i, :, :, :].data.cpu())
            all_predictions.append(pred_arg[i, :, :].data.cpu())
            all_targets.append(masks[i, :, :].data.cpu())

        all_image_ids += batch['imageIDs']

    all_logits = torch.stack(all_logits, dim=0)
    all_predictions = torch.stack(all_predictions, dim=0)
    all_targets = torch.stack(all_targets, dim=0)
    loss_total = np.mean(loss_total)

    return all_logits, all_predictions, all_targets, all_image_ids, loss_total


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0, 1, 2, 3'

    args = get_args(argparse.ArgumentParser())
    print(args.model_name)
    print('Labels: {}'.format(args.num_labels))

    train_loader, valid_loader, test_loader = get_data(args)
    print('train_dataset len:', len(train_loader.dataset))
    print('valid_dataset len:', len(valid_loader.dataset))
    # print('test_dataset len:', len(test_loader.dataset))
    # print('test_dataset len:', len(test_loader))

    net = U_Net(img_ch=1, output_ch=args.num_labels)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net = net.cuda()

    # proxy = net.module.Conv_point.weight.clone().detach()
    # print(proxy.shape)

    if args.inference:
        checkpoint = torch.load(args.model_name + '/best_model.pt')
        net.load_state_dict(checkpoint['state_dict'])
        print('model loaded')
        print("epoch:", checkpoint['epoch'])
        print("valid_mDSC:", checkpoint['valid_mDSC'])

        model_inference(args, net, valid_loader)
    else:
        model_train(args, net, train_loader, valid_loader, test_loader)
