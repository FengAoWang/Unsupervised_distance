import torch.nn as nn
import torch
import torch.nn.functional as F


class ModelWithDistanceOutput(torch.nn.Module):
    def __init__(self, base_model, centroid):
        super(ModelWithDistanceOutput, self).__init__()
        self.base_model = base_model
        self.centroid = centroid.cuda()
        self.centroid.requires_grad_(True)
        self.omics = {'gex': 0, 'methy': 1, 'mut': 2, 'cna': 3}

    def forward(self, input_omics1, input_omics2, input_omics3, input_omics4):
        # input_omics1, input_omics2, input_omics3, input_omics4, label = input_x
        # label = label.long()
        input_omics = [input_omics1, input_omics2, input_omics3, input_omics4]
        embeddings = self.base_model.get_omics_specific_embedding(input_omics, 1, self.omics)
        # print(embeddings.shape, self.centroid[label[0]].shape)
        size = embeddings.shape[0]
        center_embedding = self.centroid.repeat(size, 1)
        center_embedding = torch.squeeze(center_embedding, dim=1)
        #distance = torch.norm(center_embedding - embeddings, dim=1)
        distance = self.get_batch_l1_distance(embeddings, center_embedding)

        # print(((embeddings - self.centroid[label[0]]) ** 2).sum(dim=1))
        return distance

    @staticmethod
    def get_cosine_distance(embeddings, center_embedding):
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        center_embedding_norm = F.normalize(center_embedding, p=2, dim=1)
        return 1 - torch.sum(embeddings_norm * center_embedding_norm, dim=1)

    @staticmethod
    def get_batch_l1_distance(embeddings, center_embeddings):
        # 计算L1距离
        l1_distance = torch.sum(torch.abs(embeddings - center_embeddings), dim=1)

        return l1_distance


class Model_Explain_DownstreamTask(torch.nn.Module):
    def __init__(self, base_model, omics):
        super(Model_Explain_DownstreamTask, self).__init__()
        self.base_model = base_model
        self.omics = omics

    def forward(self, *input_omics):

        outcome = self.base_model(list(input_omics), 1, self.omics)
        return outcome
