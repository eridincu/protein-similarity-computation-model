# -- coding: utf-8 --
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import os
import requests
import time
import json
import numpy as np
import gc
import logging
import json

from tqdm.auto import tqdm

logging.basicConfig(filename="LOGS_prot_t5_xl_bfd.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

"""<b>2. Load the vocabulary and prot_t5_xl_bfd Model<b>"""

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd", do_lower_case=False)

model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_bfd")
gc.collect()
"""<b>3. Load the model into the GPU if avilabile and switch to inference mode<b>"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model = model.eval()

"""<b>4. Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)<b>"""

with open("content/proteins_with_space.json", 'r') as p_file:
    p_json = json.load(p_file)


proteins = p_json.values()
names = p_json.keys()

proteins = [re.sub(r"[UZOB]", "X", sequence) for sequence in proteins]

ids = tokenizer.batch_encode_plus(proteins, add_special_tokens=True, padding=True)
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)
logging.info('Starting to create embeddings.')
with torch.no_grad():
    embedding = model(input_ids=input_ids,attention_mask=attention_mask)
logging.info('Embeddings are created.')
embedding = embedding.last_hidden_state.cpu().numpy()
logging.info('Embeddings are converted.')

file_path = "proteins_prot_t5_xl_bfd.json"
protein_embedding_dict = {}

logging.info('Calculating means of embeddings.')
for seq_num in range(len(embedding)):
  seq_len = (attention_mask[seq_num] == 1).sum()
  seq_emd = embedding[seq_num][:seq_len-1]
  protein_embedding_dict[names[seq_num]] = seq_emd.mean(0)
logging.info('Calculation complete.')
logging.info('Writing JSON to ' + file_path)
with open(file_path, 'w') as p_file:
  p_file.write(json.dumps(protein_embedding_dict, cls=NumpyEncoder))
logging.info('Write complete.')
