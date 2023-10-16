import pandas as pd
import gzip
import shutil
import os
import tarfile
import csv

# file = open('TCGA_Dataset/ACC/TCGA.ACC.sampleMap%2FACC_clinicalMatrix', 'r')
# data = file.readlines()
#
# print(len(data))
# # print(data[0])
# for one_line in data:
#     # one_line = one_line.split(' ')
#     print(len(one_line), one_line)

# gzip_file_path = 'TCGA_Dataset/BRCA/mc3_gene_level%25FBRCA_mc3_gene_level.txt.gz'
# output_path = 'TCGA_Dataset/BRCA/gene_mutation'

# print(os.listdir('TCGA_Dataset/BRCA'))


# cancer_types = os.listdir('TCGA_Dataset')
# for cancer in cancer_types:
#     cancer_path = os.path.join('TCGA_Dataset', cancer)
#     file_list = os.listdir(cancer_path)
#     for file in file_list:
#         if file.endswith('gz'):
#             file_path = os.path.join(cancer_path, file)
#             out_path = os.path.join(cancer_path, file.replace('.gz', ''))
#             try:
#                 with gzip.open(file_path, 'rb') as f_in:
#                     with open(out_path, 'wb') as f_out:
#                         shutil.copyfileobj(f_in, f_out)
#             except Exception as e:
#                 print('file exists', e)

# file_path = 'TCGA_Dataset/BRCA/mc3_gene_level%252FBRCA_mc3_gene_level.txt'
# output_path = file_path.replace('.txt', '.csv')


def handle_txt_to_csv(input_file_path, output_file_path, transformed=True):
    # 输入和输出文件的路径

    # 读取 TXT 文件
    with open(input_file_path, 'r', newline='', encoding='utf-8') as infile:
        tsv_reader = csv.reader(infile, delimiter='\t')
        data = [row for row in tsv_reader]

    # 转置数据
    if transformed:
        transposed_data = list(zip(*data))
    else:
        transposed_data = data

    # 写入 CSV 文件
    with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile, delimiter=',')
        for row in transposed_data:
            csv_writer.writerow(row)


# cancer_types = os.listdir('TCGA_Dataset')
# for cancer in cancer_types:
#     cancer_path = os.path.join('TCGA_Dataset', cancer)
#     file_list = os.listdir(cancer_path)
#     for file in file_list:
#         if file.endswith('clinicalMatrix'):
#             file_path = os.path.join(cancer_path, file)
#             output_path = os.path.join(cancer_path, file + '.csv')
#             print(file_path, '  done')
#             try:
#                 handle_txt_to_csv(file_path, output_path)
#             except Exception as e:
#                 print('file exists', e)


omics_files = [
    'TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Expression_230109_modified.csv',
    'TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Methylation_230109_modified.csv',
    'TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_Mutation_230109_modified.csv',
    'TCGA_PanCancer_Data_cleaned/TCGA_PanCancerAtlas_CNA_230109_modified.csv'
]

expression_data = pd.read_csv(omics_files[0])
expression_features = expression_data.columns.values
expression_features = [expression_feature[4:] for expression_feature in expression_features][1:]

# TCGA_expression_data = pd.read_csv('TCGA_Dataset/BRCA/TCGA.BRCA.sampleMap%252FHiSeqV2_c)
# print(TCGA_expression_data.shape)
from typing import List


def select_features_from_csv(file_data: pd.DataFrame, threshold_percent: float = 1.0) -> List[str]:

    feature_columns = file_data.columns[1:]

    # Calculate the number of 1s in each column (feature)
    count_of_ones = file_data[feature_columns].apply(lambda x: x.sum())

    # Calculate the total number of samples
    total_samples = len(file_data)

    # Calculate the threshold count based on the percentage
    threshold_count = total_samples * (threshold_percent / 100)

    # Select the features that meet the threshold
    selected_features = count_of_ones[count_of_ones >= threshold_count]

    return selected_features.index.tolist()


def select_features_by_mean(df: pd.DataFrame, threshold_mean: float = 0, threshold_sd: float = 1.0) -> List[str]:
    """
    Select features that have a mean greater than `threshold_mean` from a DataFrame.
    The first column is assumed to be sample identifiers and is ignored in calculations.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        threshold_mean (float): The mean threshold for selecting features. Default is 1.0.

    Returns:
        List[str]: A list of selected feature names.
        :param df:
        :param threshold_mean:
        :param threshold_sd:
    """

    # Exclude the first column (sample identifiers) from calculations
    feature_columns = df.columns[1:]

    # Calculate the mean of each feature column
    feature_sd = df[feature_columns].apply(lambda x: x.std())
    feature_means = df[feature_columns].apply(lambda x: x.mean())

    # Select the features that meet the threshold
    selected_features = feature_sd[(feature_means > threshold_mean) & (feature_sd > threshold_sd)]

    return selected_features.index.tolist()


all_data = pd.DataFrame([])
cancer_types = os.listdir('TCGA_Dataset')
for cancer in cancer_types:
    cancer_path = os.path.join('TCGA_Dataset', cancer)
    file_list = os.listdir(cancer_path)
    for file in file_list:
        if file.endswith('PANCAN.csv'):
            file_path = os.path.join(cancer_path, file)
            try:
                file_data = pd.read_csv(file_path)
                all_data = pd.concat([all_data, file_data])
                print(cancer, file_data.shape)

            except Exception as e:
                print('file exists', e)


select_gene_expression_features = select_features_by_mean(all_data)
select_gene_expression_features = [all_data.columns.tolist()[0]] + select_gene_expression_features
all_data = all_data.loc[:, select_gene_expression_features]
all_data.to_csv('TCGA_PanCancerNormal_Data/TCGA_PanCancer_Expression.csv')
print(len(select_gene_expression_features))

