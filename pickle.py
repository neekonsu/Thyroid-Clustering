get_ipython().system('pip install keras scikit-learn scipy pandas matplotlib tqdm numpy tensorflow')
# for stats utils use sklearn
from sklearn import preprocessing as ppr
# for .gz extraction use gzip
import gzip
# for shell utils use shutil
import shutil
# for garbage collection use gc
import gc
# for dataframe use pandas
import pandas as pd
# for basic plots use matplotlib
import matplotlib.pyplot as plt
# for interfacing with filesystem use os
import os
# for execution progress bars use tqdm
from tqdm import tqdm
# for math utils use numpy
import numpy as np
# FORMAT MATPLOTLIB
#get_ipython().run_line_magic('matplotlib', 'inline')
# Force garbage collector (free unneeded RAM)
gc.collect()
get_ipython().system('wget "https://storage.googleapis.com/gtex_analysis_v7/rna_seq_data/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct.gz"')
# collect garbage
# gc.collect()


# ## 1.4 Unzip source

# In[ ]:


# open data archive with gzip
with gzip.open('./GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct.gz', 'rb') as f_in:
    # open constituant data in write mode
    with open('./GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct', 'wb') as f_out:
        # copy extracted file out of archive
        shutil.copyfileobj(f_in, f_out)
# collect garbage
# gc.collect()
# remove archive
#get_ipython().system('rm "/srv/gsfs0/projects/snyder/neekonsu/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct.gz"')
# collect garbage
# gc.collect()


# # 2 Sanitize data

# ## 2.1 Clear metadata

# In[ ]:


# read all lines except first three >(write to temporary file)>(overwrite original file with temporary)
#get_ipython().system('sed -i 1,2d "/srv/gsfs0/projects/snyder/neekonsu/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct"')
# read first line to check that datafile begins with column-names
# !head -c1k "/srv/gsfs0/projects/snyder/neekonsu/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct"


# ## 2.2 Load source to pandas dataframe

# In[ ]:


# collect garbage
# gc.collect()
# read GTEx RNAseq (TPM) to dataframe <data>
data = pd.read_csv('GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct', sep='\t', skiprows=2)
data.to_pickle("./database.pkl")
