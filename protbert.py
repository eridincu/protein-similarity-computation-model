import re
import transformers
from transformers import AutoTokenizer, AutoModel

protein_tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
protbert = AutoModel.from_pretrained('Rostlab/prot_bert')
        

@lru_cache(maxsize=1024)
def get_protbert_embedding(aa_sequence: str):
   cleaned_sequence = re.sub(r'[UZOB]', 'X', aa_sequence)
   tokens = protein_tokenizer(cleaned_sequence, return_tensors='pt')
   output = protbert(**tokens)
   return output.last_hidden_state.detach().numpy().mean(axis=1)
