# oiginal author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from scipy.io import mmread
import os
import pandas as pd
import numpy as np

def load_10X(data_dir, sparse=True, gene_labels="symbol"):
    if sparse:
        data_file = os.path.join(data_dir, "matrix.mtx")
        data = mmread(data_file).T.tocsr()
    else:
        data_file = os.path.join(data_dir, "matrix.tsv")
        data = pd.read_csv(data_file, sep="\t", index_col=0)
        data = data.T

    genes_file = os.path.join(data_dir, "genes.tsv")
    gene_info = pd.read_csv(genes_file, sep="\t", header=None, index_col=0)
    if gene_labels == "symbol":
        gene_names = gene_info[1].values
    else:
        gene_names = gene_info.index.values.astype(str)

    barcodes_file = os.path.join(data_dir, "barcodes.tsv")
    barcodes = pd.read_csv(barcodes_file, sep="\t", header=None)[0].values.astype(str)

    return {"data": data, "gene_names": gene_names, "barcodes": barcodes}
