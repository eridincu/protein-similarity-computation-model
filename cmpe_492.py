# -*- coding: utf-8 -*-
"""CMPE 492.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_NyD2sntPLPA7g0-ltrEI1Jjj61rC5ve

# **Machine Learning for Protein Similarity Computation**
"""

import io
import json
import logging
import random
import re

# import lightgbm
import numpy as np
import pandas as pd
from Bio import Align
from transformers import AutoModel, AutoTokenizer
from sklearn import svm


logging.basicConfig(filename="LOGS.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


SW_SCORES_PATH = "sw_sim_matrix.csv"

PROTEIN_TOKENIZER = AutoTokenizer.from_pretrained(
    'Rostlab/prot_bert', do_lower_case=False)
PROTBERT = AutoModel.from_pretrained('Rostlab/prot_bert')


def print_data(data):
    for index, value in data.items():
        print(f"Index : {index}, Value : {value}")


def get_protein_sequences_vectorized(file_path, vectorizer):
    '''

    Returns:
      Proteins object storing <str, str>:
        - Key: name
        - Value: sequence
    '''
    p_json = {}
    with open(file_path, 'r') as p_file:
        p_json = json.load(p_file)

        if p_json["is_vectorized"]:
            logging.info('Proteins already vectorized.')
            return p_json["proteins"]

    proteins = p_json["proteins"]
    vectorized_proteins = {}

    logging.info('Starting vectorizing proteins.')
    for name, sequence in proteins.items():
        # converts "abc" -> " a b c ", discard first and last whitespace.
        vectorized_proteins[name] = list(vectorizer(sequence.replace("", " ")[1:-1]))

        logging.info(f'Vectorized protein {name}')

    p_json["proteins"] = vectorized_proteins
    p_json["is_vectorized"] = True

    with open(file_path, 'w') as p_file:
        logging.info(f'Saving vectorized proteins to {file_path}')
        p_file.write(json.dumps(p_json))

    logging.info("Vectorized all protein sequences.")

    return vectorized_proteins


def get_similarity_df(filename):
    df = pd.read_csv(filename)
    logging.info('Similarity matrix is obtained.')

    return df


def prepare_similarity_scores(similarity_df):
    similarity_score_dict = {}
    logging.info('Preparing similarity score dictionary.')
    for _, score_list in similarity_df.iterrows():
        c = 0
        for score in score_list[1:]:
            c += 1
            similarity_score_dict[(
                score_list[0], similarity_df.columns[c])] = score

    logging.info('Prepared similarity score dictionary.')

    return similarity_score_dict


def vectorize_data(data, vectorizer):
    vector_data = {}
    for id, seq in data:
        vector_data[id] = vectorizer(seq)

    return vector_data


def get_protbert_embedding(aa_sequence: str):
    cleaned_sequence = re.sub(r'[UZOB]', 'X', aa_sequence)
    tokens = PROTEIN_TOKENIZER(cleaned_sequence, return_tensors='pt')
    output = PROTBERT(**tokens)
    return output.last_hidden_state.detach().numpy().mean(axis=1)


def split_data(similarity_df: pd.DataFrame, train_data_size: int):
    logging.info("Splitting data...")
    plain_data = list(prepare_similarity_scores(similarity_df).items())
    random.shuffle(plain_data)

    train_data = plain_data[0:train_data_size]
    test_data = plain_data[train_data_size:]

    logging.info("Splitted data.")
    return train_data, test_data


def prepare_model_data(data, protein_sequences_vectorized):
    X = {}
    y = {}

    logging.info("Preparing model data...")
    for t in data:
        first_protein = t[0][0]
        second_protein = t[0][1]
        similarity_score = t[1]

        X[t[0]] = np.concatenate((protein_sequences_vectorized[first_protein],
                                 protein_sequences_vectorized[second_protein]), axis=1)
        X[t[0][::-1]] = np.concatenate((protein_sequences_vectorized[second_protein],
                                       protein_sequences_vectorized[first_protein]), axis=1)

        y[t[0]] = similarity_score
        y[t[0][::-1]] = similarity_score

    logging.info("Prepared model data.")
    return X, y


def train_protein_similarity_model_SVM(train_X, train_y):
    clf = svm.SVC(gamma=0.001, C=100.)

    X = np.array()
    y = np.array()

    for protein_pair, vector in train_X.items():
        X.append(vector)
        y.append(train_y[protein_pair])

    clf.fit(X, y)

    return clf


def test_model(test_X, test_y, similarity_model):
    c = 0
    for protein_pair, vector in test_X:
        prediction = similarity_model.predict(vector)

        if abs(prediction - test_y[protein_pair]) <= 0.001:
            c += 1

    print(f'{c} out of {len(test_X)} samples are predicted close to correct.')
    logging.info(f'{c} out of {len(test_X)} samples are predicted close to correct.')

    return c


similarity_df = get_similarity_df('sw_sim_matrix.csv')
protein_sequences_vectorized = get_protein_sequences_vectorized(
    'proteins.json', get_protbert_embedding)

train_data, test_data = split_data(similarity_df)

logging.info("Train preparation:")
train_X, train_y = prepare_model_data(train_data, protein_sequences_vectorized)

logging.info("\Test preparation:")
test_X, test_y = prepare_model_data(train_data, protein_sequences_vectorized)

logging.info("Dumping all model data...")

with open('train_x.json', 'w') as f:
    f.write(json.dumps(train_X))
with open('train_y.json', 'w') as f:
    f.write(json.dumps(train_y))
with open('test_x.json', 'w') as f:
    f.write(json.dumps(test_X))
with open('test_y.json', 'w') as f:
    f.write(json.dumps(test_y))

logging.info("Completed!")

similarity_model = train_protein_similarity_model_SVM(train_X, train_y)
correctly_predicted_count = test_model(test_X, test_y, similarity_model)


"""### PROTBERT"""
"""

aligner = Align.PairwiseAligner()
aligner.mode = "local"
print(aligner.algorithm)
p1 = "MSKSKCSVGLMSSVVAPAKEPNAVGPKEVELILVKEQNGVQLTSSTLTNPRQSPVEAQDRETWGKKIDFLLSVIGFAVDLANVWRFPYLCYKNGGGAFLVPYLLFMVIAGMPLFYMELALGQFNREGAAGVWKICPILKGVGFTVILISLYVGFFYNVIIAWALHYLFSSFTTELPWIHCNNSWNSPNCSDAHPGDSSGDSSGLNDTFGTTPAAEYFERGVLHLHQSHGIDDLGPPRWQLTACLVLVIVLLYFSLWKGVKTSGKVVWITATMPYVVLTALLLRGVTLPGAIDGIRAYLSVDFYRLCEASVWIDAATQVCFSLGVGFGVLIAFSSYNKFTNNCYRDAIVTTSINSLTSFSSGFVVFSFLGYMAQKHSVPIGDVAKDGPGLIFIIYPEAIATLPLSSAWAVVFFIMLLTLGIDSAMGGMESVITGLIDEFQLLHRHRELFTLFIVLATFLLSLFCVTNGGIYVFTLLDHFAAGTSILFGVLIEAIGVAWFYGVGQFSDDIQQMTGQRPSLYWRLCWKLVSPCFLLFVVVVSIVTFRPPHYGAYIFPDWANALGWVIATSSMAMVPIYAAYKFCSLPGSFREKLAYAIAPEKDRELVDRGEVRQFTLRHWLKV"
p2 = "MNRYTTIRQLGDGTYGSVLLGRSIESGELIAIKKMKRKFYSWEECMNLREVKSLKKLNHANVVKLKEVIRENDHLYFIFEYMKENLYQLIKERNKLFPESAIRNIMYQILQGLAFIHKHGFFHRDLKPENLLCMGPELVKIADFGLAREIRSKPPYTDYVSTRWYRAPEVLLRSTNYSSPIDVWAVGCIMAEVYTLRPLFPGASEIDTIFKICQVLGTPKKTDWPEGYQLSSAMNFRWPQCVPNNLKTLIPNASSEAVQLLRDMLQWDPKKRPTASQALRYPYFQVGHPLGSTTQNLQDSEKPQKGILEKAGPPPYIKPVPPAQPPAKPHTRISSRQHQASQPPLHLTYPYKAEVSRTDHPSHLQEDKPSPLLFPSLHNKHPQSKITAGLEHKNGEIKPKSRRRWGLISRSTKDSDDWADLDDLDFSPSLSRIDLKNKKRQSDDTLCRFESVLDLKPSEPVGTGNSAPTQTSYQRRDTPTLRSAAKQHYLKHSRYLPGISIRNGILSNPGKEFIPPNPWSSSGLSGKSSGTMSVISKVNSVGSSSTSSSGLTGNYVPSFLKKEIGSAMQRVHLAPIPDPSPGYSSLKAMRPHPGRPFFHTQPRSTPGLIPRPPAAQPVHGRTDWASKYASRR"
aligner.score(p1, p2)

#alignments = aligner.score(p1, p2)
#for alignment in sorted(alignments):
#    print("Score = %.1f:" % alignment.score)
#    print(alignment)
"""
