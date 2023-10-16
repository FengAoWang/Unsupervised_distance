import requests
import os
PanCancer = {
    'ACC': 0, 'BLCA': 1, 'CESC': 2, 'CHOL': 3,
    'COAD': 4, 'DLBC': 5, 'ESCA': 6, 'GBM': 7,
    'HNSC': 8, 'KICH': 9, 'KIRC': 10, 'KIRP': 11,
    'LGG': 12, 'LIHC': 13, 'LUAD': 14, 'LUSC': 15,
    'MESO': 16, 'OV': 17, 'PAAD': 18, 'PCPG': 19,
    'PRAD': 20, 'READ': 21, 'SARC': 22, 'SKCM': 23,
    'STAD': 24, 'TGCT': 25, 'THCA': 26, 'THYM': 27,
    'UCEC': 28, 'UCS': 29, 'UVM': 30, 'BRCA': 31
    }

gene_expression_url = 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.ACC.sampleMap%2FHiSeqV2_PANCAN.gz'
gene_mut_url = 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/mc3_gene_level%2FACC_mc3_gene_level.txt.gz'
phenotype_url = 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.ACC.sampleMap%2FACC_clinicalMatrix'
methylation_url = 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.ACC.sampleMap%2FHumanMethylation450.gz'

cancer_types = PanCancer.keys()
for cancer in cancer_types:
    if not os.path.exists(f'TCGA_Dataset/{cancer}'):
        os.makedirs(f'TCGA_Dataset/{cancer}')
    url = f'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.{cancer}.sampleMap%2FHumanMethylation450.gz'

    my_file = requests.get(url)
    open(f'TCGA_Dataset/{cancer}/TCGA.{cancer}.sampleMap%2FHumanMethylation450.gz', 'wb').write(my_file.content)
    print(cancer)
