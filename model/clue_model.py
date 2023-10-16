import torch.nn as nn
import torch
import torch.nn.functional as F
from util.loss_function import KL_loss, reconstruction_loss, KL_divergence
from functools import reduce


def reparameterize(mean, logvar):
    std = torch.exp(logvar / 2)
    epsilon = torch.randn_like(std).cuda()
    return epsilon * std + mean


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def un_dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        un_dfs_freeze(child)


def product_of_experts(mu_set_, log_var_set_):
    tmp = 1.
    for i in range(len(mu_set_)):
        tmp += torch.div(1, torch.exp(log_var_set_[i]) ** 2)

    poe_var = torch.div(1., tmp)
    poe_log_var = torch.log(torch.sqrt(poe_var))

    tmp = 0.
    for i in range(len(mu_set_)):
        tmp += torch.div(1., torch.exp(log_var_set_[i]) ** 2) * mu_set_[i]
    poe_mu = poe_var * tmp
    return poe_mu, poe_log_var


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim[0]),
                                     nn.BatchNorm1d(hidden_dim[0]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[0], hidden_dim[1]),
                                     nn.BatchNorm1d(hidden_dim[1]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[1], hidden_dim[2]),
                                     nn.BatchNorm1d(hidden_dim[2]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[2], latent_dim),
                                     nn.BatchNorm1d(latent_dim),
                                     # nn.Dropout(0.2),
                                     nn.ReLU())

        self.mu_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                          nn.ReLU()
                                          )
        self.log_var_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                               nn.ReLU()
                                               )

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def decode(self, latent_z):
        cross_recon_x = self.decoder(latent_z)
        return cross_recon_x

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu_predictor(x)
        log_var = self.log_var_predictor(x)
        latent_z = self.reparameterize(mu, log_var)
        # recon_x = self.decoder(latent_z)
        return latent_z, mu, log_var


class decoder(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim[2]),
                                     nn.BatchNorm1d(hidden_dim[2]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[2], hidden_dim[1]),
                                     nn.BatchNorm1d(hidden_dim[1]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[1], hidden_dim[0]),
                                     nn.BatchNorm1d(hidden_dim[0]),
                                     # nn.Dropout(0.2),
                                     nn.ReLU(),

                                     nn.Linear(hidden_dim[0], input_dim),
                                     )

    def forward(self, latent_z):
        return self.decoder(latent_z)


class Multi_omics_vae(nn.Module):
    def __init__(self, modal_num, modal_dim, latent_dim, hidden_dim, omics_data, pretrain=False):
        super(Multi_omics_vae, self).__init__()

        self.k = modal_num
        self.vae_encoders = nn.ModuleList([encoder(modal_dim[i], latent_dim, hidden_dim) for i in range(self.k)])
        self.vae_decoders = nn.ModuleList([decoder(latent_dim, modal_dim[i], hidden_dim) for i in range(self.k)])
        self.omics_data = omics_data

    def forward(self, input_x, batch_size):
        output = [[self.encoders[i](input_x[i]) for i in range(len(input_x))]]
        self_elbo_loss = self.self_elbo(output, input_x)
        return self_elbo_loss

    def self_elbo(self, input_x, input_omic):
        self_vae_elbo = 0
        # keys = omics.keys()
        for i in range(self.k):
            latent_z, mu, log_var = input_x[i]
            reconstruct_omic = self.self_decoders[i](latent_z)
            self_vae_elbo += 0.01 * KL_loss(mu, log_var, 1.0) + reconstruction_loss(input_omic[i], reconstruct_omic, 1.0,
                                                                             self.omics_data[i])
        return self_vae_elbo

    @staticmethod
    def contrastive_loss(embeddings, labels, margin=1.0, distance='cosine'):
        if distance == 'euclidean':
            distances = torch.cdist(embeddings, embeddings)
        elif distance == 'cosine':
            normed_embeddings = F.normalize(embeddings, p=2, dim=1)
            distances = 1 - torch.mm(normed_embeddings, normed_embeddings.transpose(0, 1))
        else:
            raise ValueError(f"Unknown distance type: {distance}")

        labels_matrix = labels.view(-1, 1) == labels.view(1, -1)

        positive_pair_distances = distances * labels_matrix.float()
        negative_pair_distances = distances * (1 - labels_matrix.float())

        positive_loss = positive_pair_distances.sum() / labels_matrix.float().sum()
        negative_loss = F.relu(margin - negative_pair_distances).sum() / (1 - labels_matrix.float()).sum()

        return positive_loss + negative_loss


class Clue_model(nn.Module):
    def __init__(self, modal_num, modal_dim, latent_dim, hidden_dim, omics_data, pretrain=False):
        super(Clue_model, self).__init__()

        self.k = modal_num
        self.encoders = nn.ModuleList(
            nn.ModuleList([encoder(modal_dim[i], latent_dim, hidden_dim) for j in range(self.k)]) for i in
            range(self.k))
        self.self_decoders = nn.ModuleList([decoder(latent_dim, modal_dim[i], hidden_dim) for i in range(self.k)])

        self.cross_decoders = nn.ModuleList([decoder(latent_dim, modal_dim[i], hidden_dim) for i in range(self.k)])

        #   modality-invariant representation
        self.share_encoder = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                           nn.BatchNorm1d(latent_dim),
                                           nn.ReLU())
        #   modal align
        self.discriminator = nn.Sequential(nn.Linear(latent_dim, 16),
                                           nn.BatchNorm1d(16),
                                           nn.ReLU(),
                                           nn.Linear(16, modal_num))

        #   infer modal and real modal align
        self.infer_discriminator = nn.ModuleList(nn.Sequential(nn.Linear(latent_dim, 16),
                                                               nn.BatchNorm1d(16),
                                                               nn.ReLU(),
                                                               nn.Linear(16, 2))
                                                 for i in range(self.k))
        #   loss function hyperparameter
        self.omics_data = omics_data

        if pretrain:
            dfs_freeze(self.encoders)

    def forward(self, input_x, batch_size):
        output = [[self.encoders[i][j](input_x[i]) for j in range(len(input_x))] for i in range(len(input_x))]
        share_representation = self.share_representation(output)

        return output, share_representation

    def compute_generate_loss(self, input_x, batch_size):
        output = [[self.encoders[i][j](input_x[i]) for j in range(len(input_x))] for i in range(len(input_x))]

        self_elbo = self.self_elbo([output[i][i] for i in range(len(input_x))], input_x)
        cross_elbo, cross_infer_dsc_loss = self.cross_elbo(output, input_x, batch_size)
        cross_infer_loss = self.cross_infer_loss(output)
        dsc_loss = self.adversarial_loss(batch_size, output)
        generate_loss = self_elbo + cross_elbo + cross_infer_loss * cross_infer_loss - dsc_loss * 0.1 - cross_infer_dsc_loss * 0.1
        # generate_loss = self_elbo
        return generate_loss, self_elbo, cross_elbo, cross_infer_loss, dsc_loss

    def compute_dsc_loss(self, input_x, batch_size):
        output = [[self.encoders[i][j](input_x[i]) for j in range(len(input_x))] for i in range(len(input_x))]
        cross_elbo, cross_infer_dsc_loss = self.cross_elbo(output, input_x, batch_size)
        dsc_loss = self.adversarial_loss(batch_size, output)
        return cross_infer_dsc_loss, dsc_loss

    def share_representation(self, output):
        share_features = [self.share_encoder(output[i][i][1]) for i in range(self.k)]
        return share_features

    def self_elbo(self, input_x, input_omic):
        self_vae_elbo = 0
        # keys = omics.keys()
        for i in range(self.k):
            latent_z, mu, log_var = input_x[i]
            reconstruct_omic = self.self_decoders[i](latent_z)
            self_vae_elbo += 0.01 * KL_loss(mu, log_var, 1.0) + reconstruction_loss(input_omic[i], reconstruct_omic, 1.0,
                                                                             self.omics_data[i])
        return self_vae_elbo

    def cross_elbo(self, input_x, input_omic, batch_size):
        cross_elbo = 0
        cross_infer_loss = 0
        cross_modal_KL_loss = 0
        cross_modal_dsc_loss = 0

        for i in range(len(input_omic)):
            real_latent_z, real_mu, real_log_var = input_x[i][i]
            mu_set = []
            log_var_set = []
            for j in range(len(input_omic)):
                if i != j:
                    latent_z, mu, log_var = input_x[j][i]
                    mu_set.append(mu)
                    log_var_set.append(log_var)
            poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
            poe_latent_z = reparameterize(poe_mu, poe_log_var)
            reconstruct_omic = self.self_decoders[i](poe_latent_z)

            cross_elbo += 0.01 * KL_loss(poe_mu, poe_log_var, 1.0) + reconstruction_loss(input_omic[i], reconstruct_omic, 1.0,
                                                                                  self.omics_data[i])
            cross_infer_loss += reconstruction_loss(real_mu, poe_mu, 1.0, 'gaussian')

            cross_modal_KL_loss += KL_divergence(poe_mu, real_mu, poe_log_var, real_log_var)

            real_modal = torch.tensor([1 for j in range(batch_size)]).cuda()
            infer_modal = torch.tensor([0 for j in range(batch_size)]).cuda()
            pred_real_modal = self.infer_discriminator[i](real_mu)
            pred_infer_modal = self.infer_discriminator[i](poe_mu)

            cross_modal_dsc_loss += F.cross_entropy(pred_real_modal, real_modal, reduction='none')
            cross_modal_dsc_loss += F.cross_entropy(pred_infer_modal, infer_modal, reduction='none')

        cross_modal_dsc_loss = cross_modal_dsc_loss.sum(0) / (self.k * batch_size)
        return cross_elbo + cross_infer_loss + 0.01 * cross_modal_KL_loss, cross_modal_dsc_loss

    def cross_infer_loss(self, input_x):
        latent_mu = [input_x[i][i][1] for i in range(len(input_x))]
        infer_loss = 0
        for i in range(len(input_x)):
            for j in range(len(input_x)):
                if i != j:
                    latent_z_infer, latent_mu_infer, _ = input_x[j][i]
                    infer_loss += reconstruction_loss(latent_mu_infer, latent_mu[i], 1.0, 'gaussian')
        return infer_loss / self.k

    def adversarial_loss(self, batch_size, output):
        dsc_loss = 0
        for i in range(self.k):
            latent_z, mu, log_var = output[i][i]
            shared_fe = self.share_encoder(mu)

            real_modal = (torch.tensor([i for j in range(batch_size)])).cuda()
            pred_modal = self.discriminator(shared_fe)
            # print(i, pred_modal)
            dsc_loss += F.cross_entropy(pred_modal, real_modal, reduction='none')

        dsc_loss = dsc_loss.sum(0) / (self.k * batch_size)
        return dsc_loss

    def latent_z(self, input_x, omics):
        input_len = len(input_x)
        output = [[self.encoders[i][j](input_x[i]) for j in range(input_len)] for i in range(input_len)]
        embedding_list = []
        keys = list(omics.keys())
        for i in range(self.k):
            latent_z, mu, log_var = output[omics[keys[i]]][i]
            embedding_list.append(mu)
        embedding_tensor = torch.cat(embedding_list, dim=1)
        return embedding_tensor

    def get_embedding(self, input_x, batch_size, omics):
        output, share_representation = self.forward(input_x, batch_size)
        embedding_tensor = []
        keys = list(omics.keys())
        share_features = [share_representation[omics[key]] for key in keys]
        share_features = sum(share_features) / len(keys)

        for i in range(self.k):
            mu_set = []
            log_var_set = []
            for j in range(len(omics)):
                latent_z, mu, log_var = output[omics[keys[j]]][i]
                if i != j:
                    mu_set.append(mu)
                    log_var_set.append(log_var)
            poe_mu, poe_log_var = product_of_experts(mu_set, log_var_set)
            _, omic_mu, omic_log_var = output[i][i]
            poe_latent_z = reparameterize(poe_mu, poe_log_var)
            joint_mu = (omic_mu + poe_mu) / 2

            embedding_tensor.append(joint_mu)

        # embedding_tensor = self.lmf_fusion(embedding_tensor)
        embedding_tensor = torch.cat(embedding_tensor, dim=1)
        multi_representation = torch.concat((embedding_tensor, share_features), dim=1)
        return multi_representation

    def get_omics_specific_embedding(self, input_x, batch_size, omics):
        output, share_representation = self.forward(input_x, batch_size)
        embedding_tensor = []
        keys = list(omics.keys())
        for i in range(self.k):
            latent_z, mu, log_var = output[i][i]
            embedding_tensor.append(mu)
        multi_embedding = torch.concat(embedding_tensor, dim=1)
        return multi_embedding

    @staticmethod
    def contrastive_loss(embeddings, labels, margin=1.0, distance='cosine'):

        if distance == 'euclidean':
            distances = torch.cdist(embeddings, embeddings)
        elif distance == 'cosine':
            normed_embeddings = F.normalize(embeddings, p=2, dim=1)
            distances = 1 - torch.mm(normed_embeddings, normed_embeddings.transpose(0, 1))
        else:
            raise ValueError(f"Unknown distance type: {distance}")

        labels_matrix = labels.view(-1, 1) == labels.view(1, -1)

        positive_pair_distances = distances * labels_matrix.float()
        negative_pair_distances = distances * (1 - labels_matrix.float())

        positive_loss = positive_pair_distances.sum() / labels_matrix.float().sum()
        negative_loss = F.relu(margin - negative_pair_distances).sum() / (1 - labels_matrix.float()).sum()

        return positive_loss + negative_loss

