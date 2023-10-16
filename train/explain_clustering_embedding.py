import sys
sys.path.append('/home/wfa/project/Unsupervised_distance')
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from model.explain_model import ModelWithDistanceOutput
from model.clue_model import Clue_model
from dataset.dataset import CancerDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from captum.attr import IntegratedGradients, DeepLift
import pandas as pd



PanCancer = {
    'ACC': 0, 'BLCA': 1, 'CESC': 2, 'CHOL': 3,
    'COAD': 4, 'DLBC': 5, 'ESCA': 6, 'GBM': 7,
    'HNSC': 8, 'KICH': 9, 'KIRC': 10, 'KIRP': 11,
    'LGG': 12, 'LIHC': 13, 'LUAD': 14, 'LUSC': 15,
    'MESO': 16, 'OV': 17, 'PAAD': 18, 'PCPG': 19,
    'PRAD': 20, 'READ': 21, 'SARC': 22, 'SKCM': 23,
    'STAD': 24, 'TGCT': 25, 'THCA': 26, 'THYM': 27,
    'UCEC': 28, 'UCS': 29, 'UVM': 30, 'BRCA': 31,

    }
#
#
# embeddings = torch.load('../model/model_dict/dim16/TCGA_pancancer_multi_train_embedding_fold1_epoch9.pt', map_location='cpu')
# labels = torch.load('../model/model_dict/dim16/TCGA_pancancer_train_fold1_epoch9_all_label.pt', map_location='cpu').tolist()
#
#
# cluster_centers = {}
# embeddings_by_cancer_type = {}
# for embedding, label in zip(embeddings, labels):
#     if label not in embeddings_by_cancer_type.keys():
#         embeddings_by_cancer_type[label] = []
#     embeddings_by_cancer_type[label].append(embedding.detach().numpy())
#
# # print(embeddings_by_cancer_type.keys(), embeddings_by_cancer_type.values())
# for label, embeddings in embeddings_by_cancer_type.items():
#     kmeans = KMeans(n_clusters=1, random_state=0).fit(embeddings)
#     cluster_centers[label] = kmeans.cluster_centers_[0]
#
# # Sort the cluster centers by their keys (cancer types)
# sorted_cluster_centers = sorted(cluster_centers.items(), key=lambda x: x[0])
# print(sorted_cluster_centers)
# # Extract the cluster centers and concatenate them
# concat_cluster_centers = torch.concat([torch.unsqueeze(torch.tensor(center), dim=0) for _, center in sorted_cluster_centers], dim=0)
#
# # Save the concatenated cluster centers to a pt file
# torch.save(concat_cluster_centers, 'concat_cluster_centers.pt')



omics_files = [
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Expression_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Methylation_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Mutation_230109_modified.csv',
    '../data/TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_CNA_230109_modified.csv'
]

omics_data = [pd.read_csv(file_path) for file_path in omics_files]

features = [omics.columns[1:] for omics in omics_data]

clinical_file = '../data/TCGA_PanCancer_Data_cleaned/cleaned_clinical_info.csv'

train_index_path = '../data/TCGA_PanCancer_Data_cleaned/train_data.csv'
test_index_path = '../data/TCGA_PanCancer_Data_cleaned/test_data.csv'

# omics_data_type = ['gaussian', 'gaussian']
omics_data_type = ['gaussian', 'gaussian', 'gaussian', 'gaussian']

# omics_data_type = ['bernoulli', 'bernoulli', 'bernoulli', 'bernoulli']
# cancers = ['KIRP', 'KIRC', 'KICH', 'LGG', 'GBM', 'COAD', 'STAD', 'PAAD', 'ESCA', 'BRCA', 'CESC', 'UCEC', 'UCS', 'LUAD', 'LUSC']
cancers = ['BLCA', 'HNSC', 'LIHC', 'SKCM', 'PRAD', 'LGG', 'STAD', 'PAAD', 'BRCA', 'LUAD', 'KIRC', 'KIRP', 'UCEC']

# cancer = 'KIRP'

torch.cuda.set_device(7)


model = Clue_model(4, [6016, 6617, 4539, 7460], 16, [2048, 1024, 512], omics_data_type)
model_dict = torch.load('../model/model_dict/dim16/TCGA_pancancer_pretrain_model_fold1.pt', map_location='cpu')
model.load_state_dict(model_dict)


def compute_gene_score(cancer):
    omics = {'gex': 0, 'methy': 1, 'mut': 2, 'cna': 3}
    embedding_centers = torch.load('concat_cluster_centers.pt', map_location='cpu')
    #
    # print(embedding_centers.shape[0])
    omics_data_type = ['gaussian', 'gaussian', 'gaussian', 'gaussian']

    model.cuda()
    model.eval()

    explain_model = ModelWithDistanceOutput(model, embedding_centers[PanCancer[cancer]])

    train_dataset = CancerDataset(omics_files, ['gex', 'methy', 'mut', 'cna'], clinical_file, train_index_path, 1, [cancer])

    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    # 初始化 Integrated Gradients
    ig = IntegratedGradients(explain_model)
    df = DeepLift(explain_model)
    # with torch.no_grad():
    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch, data in enumerate(tepoch):
            os_event, os_time, omics_data, cancer_label = data
            torch.save(os_time, f'{cancer}_os_time.pt')
            torch.save(os_event, f'{cancer}_os_event.pt')
            cancer_label = cancer_label.cuda()
            # cancer_label = cancer_label.squeeze()
            input_x = [omics_data[key].cuda() for key in omics_data.keys()]
            distance = explain_model(input_x[0], input_x[1], input_x[2], input_x[3])
            input_x = (*input_x, )
            # input_x = torch.concat(input_x, dim=0)

            torch.save(distance, f'{cancer}_embedding_distance.pt')
            attr_list = ig.attribute(input_x)

        omics1 = attr_list[2].mean(dim=0).detach().cpu().numpy()
        df = pd.DataFrame({'feature': features[2], 'importance': omics1})
        df['importance_abs'] = df['importance'].abs()
        df = df.sort_values(by='importance_abs', ascending=False)
        df.to_csv(f'{cancer}_mut_gene_scores.csv')
        top_features = df['feature'].head(10)
        print(cancer)
        print(top_features)


for cancer in cancers:
    compute_gene_score(cancer)
