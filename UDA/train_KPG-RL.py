import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
from pre_process import ImageList
import copy
import random
import tqdm
import utils
from keypointguide_POT.partial_OT import partial_ot


def image_classification_test(loader, model, tencrop=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        if not tencrop:
            for i in tqdm.trange(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            for i in tqdm.trange(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                bs, nc, c, h, w = inputs.size()
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels
                _, outputs = model(inputs.view(-1, c, h, w))
                outputs = outputs.view(bs, nc, -1).mean(1)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() ==
                         all_label).item() / float(all_label.size()[0])

    return accuracy


def train(config):
    tencrop = config["tencrop"]
    source = config["source"]
    target = config["target"]
    prep_dict = {}
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])

    prep_dict["test"] = prep.image_test(
        **config["prep"]['params'], tencrop=tencrop)

    # prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(),
                                transform=prep_dict["source"], root=config["root"])
    # source_sampler = BalancedBatchSampler(source_labels, batch_size=train_bs)
    # dset_loaders["source"] = DataLoader(
    #     dsets["source"], batch_sampler=source_sampler, num_workers=4 )
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs,
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(),
                                transform=prep_dict["target"], root=config["root"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs,
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(),
                              transform=prep_dict["test"], root=config["root"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs,
                                      shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    # set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    # add additional network for some methods
    ad_net = network.AdversarialNetworkSp(
        base_network.output_num(), 1024, radius=config["network"]["params"]["radius"])
    ad_net = ad_net.cuda()

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net)
        base_network = nn.DataParallel(base_network)

    parameter_classifier = [base_network.get_parameters()[2]]
    parameter_feature = base_network.get_parameters()[
        0:2] + ad_net.get_parameters()

    # set optimizer
    optimizer_config = config["optimizer"]
    optimizer_classfier = optimizer_config["type"](parameter_classifier,
                                                   **(optimizer_config["optim_params"]))
    optimizer_feature = optimizer_config["type"](parameter_feature,
                                                 **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer_feature.param_groups:
        param_lr.append(param_group["lr"])
    param_lr.append(optimizer_classfier.param_groups[0]["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    # train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    best_model = copy.deepcopy(base_network)

    Cs_memory = torch.zeros(class_num, 256).cuda()
    Ct_memory = torch.zeros(class_num, 256).cuda()
    for i in range(config["iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = image_classification_test(
                dset_loaders, base_network, tencrop=tencrop)
            temp_model = base_network
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = copy.deepcopy(temp_model)
            log_str = "iter: {:05d},\ttemp_acc:{:.4f},\t best_acc: {:.4f}" \
                "".format(i, temp_acc, best_acc)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)
        if (i + 1) % config["snapshot_interval"] == 0:
            if not os.path.exists("save/"):
                os.makedirs("save/")
            torch.save(best_model, 'save/' + source + '_' + target + '.pkl')

        # train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer_classfier = lr_scheduler(
            optimizer_classfier, i, **schedule_param)
        optimizer_feature = lr_scheduler(
            optimizer_feature, i, **schedule_param)

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = next(iter_source)
        inputs_target, labels_target = next(iter_target)
        inputs_source, inputs_target, labels_source = inputs_source.cuda(
        ), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)  # (B, 256) (B, 65)
        features_target, outputs_target = base_network(inputs_target)
        # features = torch.cat((features_source, features_target), dim=0)
        pseu_labels_target = torch.argmax(outputs_target, dim=1)
        Cs_memory, Ct_memory = loss.update_center(features_source, features_target, labels_source, pseu_labels_target,
                                                  Cs_memory, Ct_memory)  # (65, 256) (65, 256)
        feats = torch.cat([Cs_memory, features_source], dim=0)  # (125, 256)
        featt = torch.cat([Ct_memory, features_target], dim=0)  # (125, 256)
        I = np.arange(len(Cs_memory)).tolist() # (65)
        J = I
        C1 = loss.cosine_matrix(feats, Cs_memory.data)  # (125, 65)
        C2 = loss.cosine_matrix(featt, Ct_memory.data)
        R = utils.structure_metrix_relation(C1, C2, tau=0.1)  # (125, 125)
        C = loss.cosine_matrix(feats, featt)  # (125, 125)
        alpha = 0.5
        C = alpha*C + (1.0-alpha)*R
        C_cpu = C.cpu().data
        p = torch.ones(len(C1)) / len(C1)
        q = torch.ones(len(C1)) / len(C1)
        pi = partial_ot(p, q, C_cpu, I, J, s=0.6)
        transfer_loss = len(Cs_memory)*torch.sum(pi.cuda()*C)

        gamma = network.calc_coeff(i)
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        loss_total = classifier_loss + gamma*transfer_loss

        optimizer_classfier.zero_grad()
        optimizer_feature.zero_grad()

        loss_total.backward()
        optimizer_feature.step()
        optimizer_classfier.step()

        print('step:{: d},\t,class_loss:{:.4f},\t,R_loss:{:.4f}'
              ''.format(i, classifier_loss.item(), transfer_loss.item()))
        Cs_memory.detach_()
        Ct_memory.detach_()
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for RSDA-DANN')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--source', type=str, default='Art')
    parser.add_argument('--target', type=str, default='Clipart')
    parser.add_argument('--test_interval', type=int, default=50,
                        help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help="interval of two continuous output model")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--stages', type=int, default=5,
                        help="training stages")
    parser.add_argument('--radius', type=float, default=10, help="radius")
    args = parser.parse_args()
    root = "/home/icml007/Nightmare4214/datasets/OfficeHomeDataset_10072016"
    # /home/icml007/Nightmare4214/datasets/OfficeHomeDataset_10072016
    # s_dset_path = '{}/office-home/'.format(root) + args.source + '.txt'
    s_dset_path = f'{root}/{args.source}.txt'
    # t_dset_path = '{}/office-home/'.format(root) + args.target + '.txt'
    t_dset_path = f'{root}/{args.target}.txt'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    config = {}
    config["source"] = args.source
    config["target"] = args.target
    config["tencrop"] = True
    config["gpu"] = args.gpu_id
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/"
    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    config["out_file"] = open(
        osp.join(config["output_path"], args.source+"_"+args.target + "_log.txt"), "w")

    config["prep"] = {'params': {"resize_size": 256, "crop_size": 224}}
    config["network"] = {"name": network.ResNetCos,
                         "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True, "class_num": 65, "radius": args.radius}}
    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9,
                           "weight_decay": 0.0005, "nesterov": True}, "lr_type": "inv",
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}
    config["data"] = {"source": {"list_path": s_dset_path, "batch_size": 65},
                      "target": {"list_path": t_dset_path, "batch_size": 65},
                      "test": {"list_path": t_dset_path, "batch_size": 72}}
    config["out_file"].flush()
    config["iterations"] = 5000
    config["root"] = root
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    config["out_file"].write('\n--- initialization ---\n')
    best_acc = train(config)  # This accaracy is the result of method "DANN+S"
