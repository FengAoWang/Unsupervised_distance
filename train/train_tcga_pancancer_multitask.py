import os
import sys
sys.path.append('/home/wfa/project/Unsupervised_distance')
import torch
from dataset.dataset import CancerDataset
from model.clue_model import Clue_model, dfs_freeze, un_dfs_freeze
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from util.loss_function import cox_loss, c_index
from lifelines.utils import concordance_index
import pickle
import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from LogME import LogME
logme = LogME(regression=False)


def set_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(66)

omics_files = [
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Expression_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Methylation_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Mutation_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_CNA_230109_modified.csv'
]

clinical_file = '../data/TCGA_PanCancer_Data_cleaned/cleaned_clinical_info.csv'

train_index_path = '../data/TCGA_PanCancer_Data_cleaned/train_data.csv'
test_index_path = '../data/TCGA_PanCancer_Data_cleaned/test_data.csv'

# omics_data_type = ['gaussian', 'gaussian']
omics_data_type = ['gaussian', 'gaussian', 'gaussian', 'gaussian']

# omics_data_type = ['bernoulli', 'bernoulli', 'bernoulli', 'bernoulli']

omics = {'gex': 0, 'methy': 1, 'mut': 2, 'cna': 3}
# omics = {'gex': 0, 'methy': 1}

latent_z_dim = 16


#   pretrain
def train_pretrain(train_dataloader, model, epoch, cancer, optimizer, dsc_optimizer, fold):
    model.train()
    # model = model.state_dict()
    print(f'-----start epoch {epoch} training-----')
    total_loss = 0
    total_self_elbo = 0
    total_cross_elbo = 0
    total_cross_infer_loss = 0
    total_dsc_loss = 0
    total_ad_loss = 0
    total_cross_infer_dsc_loss = 0
    Loss = []
    pancancer_embedding = torch.Tensor([]).cuda()
    all_label = torch.Tensor([]).cuda()

    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            tepoch.set_description(f" Epoch {epoch}: ")
            os_time, os_event, omics_data, cancer_label = data
            cancer_label = cancer_label.cuda()
            cancer_label = cancer_label.squeeze()
            all_label = torch.concat((all_label, cancer_label), dim=0)

            input_x = []
            for key in omics_data.keys():
                omic = omics_data[key]
                omic = omic.cuda()
                # print(omic)
                input_x.append(omic)
            un_dfs_freeze(model.discriminator)
            un_dfs_freeze(model.infer_discriminator)
            cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, os_event.size(0))
            ad_loss = cross_infer_dsc_loss + dsc_loss
            total_ad_loss += dsc_loss.item()

            dsc_optimizer.zero_grad()
            dsc_loss.backward(retain_graph=True)
            dsc_optimizer.step()

            dfs_freeze(model.discriminator)
            dfs_freeze(model.infer_discriminator)

            loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model.compute_generate_loss(input_x,
                                                                                                  os_event.size(0))

            total_self_elbo += self_elbo.item()
            total_cross_elbo += cross_elbo.item()
            total_cross_infer_loss += cross_infer_loss.item()
            multi_embedding = model.get_omics_specific_embedding(input_x, os_event.size(0), omics)
            pancancer_embedding = torch.concat((pancancer_embedding, multi_embedding), dim=0)
            contrastive_loss = model.contrastive_loss(multi_embedding, cancer_label)

            loss += contrastive_loss

            optimize_loss = self_elbo + contrastive_loss

            # loss = ce_loss
            total_dsc_loss += dsc_loss.item()
            total_loss += loss.item()
            optimizer.zero_grad()
            optimize_loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item(), self_elbo_loss=self_elbo.item(), cross_elbo_loss=cross_elbo.item(),
                               cross_infer_loss=cross_infer_loss.item(), dsc_loss=dsc_loss.item())

        print('total loss: ', total_loss / len(train_dataloader))
        Loss.append(total_loss / len(train_dataloader))
        print('self elbo loss: ', total_self_elbo / len(train_dataloader))
        Loss.append(total_self_elbo / len(train_dataloader))
        print('cross elbo loss: ', total_cross_elbo / len(train_dataloader))
        Loss.append(total_cross_elbo / len(train_dataloader))
        print('cross infer loss: ', total_cross_infer_loss / len(train_dataloader))
        Loss.append(total_cross_infer_loss / len(train_dataloader))
        print('ad loss', total_ad_loss / len(train_dataloader))
        print('dsc loss', total_dsc_loss / len(train_dataloader))
        Loss.append(total_dsc_loss / len(train_dataloader))

        torch.save(pancancer_embedding, f'../train_log//dim{latent_z_dim}/TCGA_pancancer_multi_train_embedding_fold{fold}_epoch{epoch}.pt')
        torch.save(all_label, f'../train_log//dim{latent_z_dim}/TCGA_pancancer_train_fold{fold}_epoch{epoch}_all_label.pt')

        pretrain_score = logme.fit(pancancer_embedding.detach().cpu().numpy(), all_label.cpu().numpy())
        print('pretrain score:', pretrain_score)
        return Loss


def val_pretrain(test_dataloader, model, epoch, cancer, fold):
    model.eval()
    # model = model.state_dict()
    print(f'-----start epoch {epoch} val-----')
    total_loss = 0
    total_self_elbo = 0
    total_cross_elbo = 0
    total_cross_infer_loss = 0
    total_dsc_loss = 0
    total_cross_infer_dsc_loss = 0
    Loss = []
    pancancer_embedding = torch.Tensor([]).cuda()
    all_label = torch.Tensor([]).cuda()
    with torch.no_grad():
        with tqdm(test_dataloader, unit='batch') as tepoch:
            for batch, data in enumerate(tepoch):
                tepoch.set_description(f" Epoch {epoch}: ")
                os_time, os_event, omics_data, cancer_label = data
                cancer_label = cancer_label.cuda()
                cancer_label = cancer_label.squeeze()
                all_label = torch.concat((all_label, cancer_label), dim=0)
                input_x = []
                for key in omics_data.keys():
                    omic = omics_data[key]
                    omic = omic.cuda()
                    # print(omic.size())
                    input_x.append(omic)

                cross_infer_dsc_loss, dsc_loss = model.compute_dsc_loss(input_x, os_event.size(0))

                total_cross_infer_dsc_loss += cross_infer_dsc_loss.item()

                loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss = model.compute_generate_loss(input_x,
                                                                                                      os_event.size(0))
                multi_embedding = model.get_embedding(input_x, os_event.size(0), omics)

                pancancer_embedding = torch.concat((pancancer_embedding, multi_embedding), dim=0)
                total_self_elbo += self_elbo.item()
                total_cross_elbo += cross_elbo.item()
                total_cross_infer_loss += cross_infer_loss.item()

                total_dsc_loss += dsc_loss.item()
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item(), self_elbo_loss=self_elbo.item(), cross_elbo_loss=cross_elbo.item(),
                                   cross_infer_loss=cross_infer_loss.item(), dsc_loss=dsc_loss.item())

            print('test total loss: ', total_loss / len(test_dataloader))
            Loss.append(total_loss / len(test_dataloader))
            print('test self elbo loss: ', total_self_elbo / len(test_dataloader))
            Loss.append(total_self_elbo / len(test_dataloader))
            print('test cross elbo loss: ', total_cross_elbo / len(test_dataloader))
            Loss.append(total_cross_elbo / len(test_dataloader))
            print('test cross infer loss: ', total_cross_infer_loss / len(test_dataloader))
            Loss.append(total_cross_infer_loss / len(test_dataloader))
            print('test ad loss', total_cross_infer_dsc_loss / len(test_dataloader))
            print('test dsc loss', total_dsc_loss / len(test_dataloader))
            Loss.append(total_dsc_loss / len(test_dataloader))
            torch.save(pancancer_embedding,
                       f'../train_log/dim{latent_z_dim}/TCGA_pancancer_multi_test_embedding_fold{fold}_epoch{epoch}.pt')
            torch.save(all_label, f'../train_log/dim{latent_z_dim}/TCGA_pancancer_test_fold{fold}_epoch{epoch}_all_label.pt')
    return Loss


def TCGA_Dataset_pretrain(fold, epochs, device_id, cancer_types=None):
    train_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, train_index_path,
                                  fold + 1)
    test_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, test_index_path, fold + 1)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    model = Clue_model(4, [6016, 6617, 4539, 7460], latent_z_dim, [2048, 1024, 512], omics_data_type)
    torch.cuda.set_device(device_id)
    model.cuda()
    print(len(train_dataset))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    dsc_parameters = list(model.discriminator.parameters()) + list(model.infer_discriminator.parameters())
    dsc_optimizer = torch.optim.Adam(dsc_parameters, lr=0.0001)

    Loss_list = []
    test_Loss_list = []
    for epoch in range(epochs):

        start_time = time.time()
        Loss_list.append(train_pretrain(train_dataloader, model, epoch, 'PanCancer', optimizer, dsc_optimizer, fold))

        test_Loss_list.append(val_pretrain(test_dataloader, model, epoch, 'PanCancer', fold))
        print(f'fold{fold} time used: ', time.time() - start_time)

    model_dict = model.state_dict()
    Loss_list = torch.Tensor(Loss_list)
    test_Loss_list = torch.Tensor(test_Loss_list)

    #torch.save(test_Loss_list, f'../model/model_dict/dim{latent_z_dim}/TCGA_pancancer_pretrain_test_loss_fold{fold}_v2.pt')
    #torch.save(Loss_list, f'../model/model_dict/dim{latent_z_dim}/TCGA_pancancer_pretrain_train_loss_fold{fold}_v2.pt')
    torch.save(model_dict, f'../model/model_dict/dim{latent_z_dim}/TCGA_pancancer_pretrain_model_fold{fold}.pt')


device_ids = [0, 3, 5, 6, 7]
folds = 5
all_epochs = 30


#   multiprocessing pretrain_fold
def multiprocessing_train_fold(function, func_args_list):
    processes = []
    for i in range(folds):
        p = mp.Process(target=function, args=func_args_list[i])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


pretrain_func_args = [(i, all_epochs, device_ids[i]) for i in range(folds)]
multiprocessing_train_fold(TCGA_Dataset_pretrain, pretrain_func_args)
