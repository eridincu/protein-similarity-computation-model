# -*- coding: utf-8 -*-
"""CMPE 492.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_NyD2sntPLPA7g0-ltrEI1Jjj61rC5ve

# **Machine Learning for Protein Similarity Computation**
"""

import json
import logging
import random
import re
import pickle
import time
import sys


import numpy as np
import pandas as pd
#from Bio import Align
#from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsCV
from sklearn.neighbors import KNeighborsRegressor
# from lightgbm import LGBMRegressor


from sklearn.model_selection import train_test_split

args = sys.argv[1:]

if len(args) == 1:
    args.append("default")

logging.basicConfig(filename="LOGS" + args[0] + "_" + args[1] + ".txt",
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

#PROTEIN_TOKENIZER = AutoTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
#PROTBERT = AutoModel.from_pretrained('Rostlab/prot_bert')


# print("Getting ProtBert Model and Tokenizer")
# PROTEIN_TOKENIZER = AutoTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
# PROTBERT = AutoModel.from_pretrained('Rostlab/prot_bert')
# PROTEIN_TOKENIZER.save_pretrained("/Users/hazelcast/Desktop/protein-similarity-computation-model/")
# PROTBERT.save_pretrained("/Users/hazelcast/Desktop/protein-similarity-computation-model/")
# print("Model and tokenizer saved in local")

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

        logging.info('Vectorized protein ' + name)

    p_json["proteins"] = vectorized_proteins
    p_json["is_vectorized"] = True

    with open(file_path, 'w') as p_file:
        logging.info('Saving vectorized proteins to ' + file_path)
        p_file.write(json.dumps(p_json, cls=NumpyEncoder))

    logging.info("Vectorized all protein sequences.")

    return vectorized_proteins


def get_similarity_df(filename):
    df = pd.read_csv(filename)
    SW_score_dict = {}
    c = 0
    for _, score_list in df.iterrows():
      c = 0
      for score in score_list[1:]:
        c = c + 1
        #print(score_list[0], df_SW_score.columns[c], score)
        SW_score_dict[(score_list[0], df.columns[c])] = score
    logging.info('Similarity matrix is obtained.')

    return SW_score_dict


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


def get_protbert_embedding(aa_sequence):
    cleaned_sequence = re.sub(r'[UZOB]', 'X', aa_sequence)
    #tokens = PROTEIN_TOKENIZER(cleaned_sequence, return_tensors='pt')
    #output = PROTBERT(**tokens)
    #return output.last_hidden_state.detach().numpy().mean(axis=1)
    return ''


def split_data(similarity_df, protein_sequences_vectorized, train_data_size):
    plain_data = list(protein_sequences_vectorized.items())
    random.shuffle(plain_data)
    train_X = plain_data[:train_data_size]
    test_X = plain_data[train_data_size:]
    logging.info("Splitting data...")

    train_X_final = []
    train_Y_final = []
    for id, vector in train_X:
        for id2, vector2 in train_X:
            train_X_final.append(np.concatenate((vector, vector2)))
            train_Y_final.append(similarity_df[(id, id2)])

    test_X_final = []
    test_Y_final = []
    for id, vector in test_X:
        for id2, vector2 in train_X:
            test_X_final.append(np.concatenate((vector, vector2)))
            test_Y_final.append(similarity_df[(id, id2)])
        for id2, vector2 in test_X:
            test_X_final.append(np.concatenate((vector, vector2)))
            test_Y_final.append(similarity_df[(id, id2)])

    for id, vector in train_X:
        for id2, vector2 in test_X:
            test_X_final.append(np.concatenate((vector, vector2)))
            test_Y_final.append(similarity_df[(id, id2)])

    logging.info("Splitted data.")
    return train_X_final, train_Y_final, test_X_final, test_Y_final


def prepare_model_data(data, protein_sequences_vectorized):
    X = []
    y = []

    logging.info("Preparing model data...")
    for t in data:
        first_protein = t[0][0]
        second_protein = t[0][1]
        similarity_score = t[1]

        X.append(np.concatenate((protein_sequences_vectorized[first_protein],
                                 protein_sequences_vectorized[second_protein])))

        y.append(similarity_score)

    logging.info("Prepared model data.")
    return X, y


def train_and_save_model(model_name, model, train_X, train_y):
    logging.info('Training model ' + model_name)
    start_time = time.time()
    model.fit(np.array(train_X), np.array(train_y))
    logging.info('Training completed in ' + str(time.time() - start_time) + ' seconds.')

    with open(model_name + '.pickle', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('Saved ' + model_name + ' model as a pickle.\n')

    return model

# def train_protein_similarity_model_LGBM(train_X, train_Y):
#     model = LGBMRegressor()

#     logging.info('Training model')
#     model.fit(np.array(train_X), np.array(train_Y))

#     # for cross validation
#     #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#     #n_scores = cross_val_score(model, np.array(train_X), np.array(train_Y), scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
#     # report performance
#     #print(n_scores)
#     #print('MAE: %.3f (%.3f)' % (np.mean(n_scores),np.std(n_scores)))


#     logging.info('Training completed.')

#     with open('LGBM.pickle', 'wb') as f:
#         pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
#         logging.info('Saved the model as a pickle.\n')

#     return model
def report_statistics(error_margin, predicted, total):
    logging.info(error_margin)
    logging.info(str(predicted) + ' out of ' + str(total) + ' samples are predicted close to correct.')
    logging.info('Accuracy: ' + str(float(predicted) / total) + "\n")
    
def test_model(model_name, test_X, test_y, similarity_model):
    logging.info('Testing ' + model_name)
    start_time = time.time()

    error_margin_count = {0.1: 0, 0.05: 0, 0.04: 0, 0.03: 0, 0.02: 0, 0.01: 0, 0.001: 0}
    for vector, actual in zip(test_X, test_y):
        prediction = similarity_model.predict(np.array(vector).reshape(1, -1))

        #print(f'Prediction: {prediction}, Actual: {actual}, difference: {abs(prediction - actual)}')
        for error_margin in error_margin_count.keys():
            if abs(prediction - actual) <= error_margin:
                error_margin_count[error_margin] += 1
    
    for error_margin, error_count in error_margin_count.items():
        report_statistics(str(error_margin), error_count, len(test_X))

    logging.info('Test completed in ' + str(time.time() - start_time) + ' seconds.\n')
    return error_margin_count


similarity_df = get_similarity_df('sw_sim_matrix.csv')
protein_sequences_vectorized = get_protein_sequences_vectorized(
    'proteins.json', get_protbert_embedding)

train_X, train_y, test_X, test_y = split_data(similarity_df, protein_sequences_vectorized, train_data_size=445)

# print(len(train_X))
# print(len(train_y))
# print(len(test_X))
# print(len(test_y))


regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

models = {
    "lasso": {
        'default': {
            "Cross Validated Lasso": LassoLarsCV(cv=5, normalize=True),
            "Cross Validated-Normalized Lasso": LassoLarsCV(cv=5, normalize=False),
        }
    },
    "svr": {
        'default': {
            "SVR, kernel=linear, c=1, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(kernel="linear", C=1.0, epsilon=0.2)),
        },
        'linear': {
            "SVR, kernel=linear, c=100, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(kernel="linear", C=100.0, epsilon=0.2, )),
            "SVR, kernel=linear, c=10, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(kernel="linear", C=10.0, epsilon=0.2)),
            "SVR, kernel=linear, c=1, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(kernel="linear", C=1.0, epsilon=0.2)),
            "SVR, kernel=linear, c=0.1, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(kernel="linear", C=0.1, epsilon=0.2)),
        },
        'rbf': {
            "SVR, kernel=rbf, c=100, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(C=100.0, epsilon=0.2, )),
            "SVR, kernel=rbf, c=10, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.2)),
            "SVR, kernel=rbf, c=1, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)),
            "SVR, kernel=rbf, c=0.1, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(C=0.1, epsilon=0.2)),
        },
        'sigmoid': {
            "SVR, kernel=sigmoid, c=100, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(kernel='sigmoid', C=100.0, epsilon=0.2)),
            "SVR, kernel=sigmoid, c=10, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(kernel='sigmoid', C=10.0, epsilon=0.2)),
            "SVR, kernel=sigmoid, c=1, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(kernel='sigmoid', C=1.0, epsilon=0.2)),
            "SVR, kernel=sigmoid, c=0.1, epsilon=0.2, Standard Scaler pipeline": make_pipeline(StandardScaler(), SVR(kernel='sigmoid', C=0.1, epsilon=0.2)),
           
        }
    },
    "linearsvr": {
        "default": {
            "LinearSVR, c=1": LinearSVR(C=1),
        },
        "no_epsilon": {
            "LinearSVR, c=0.1": LinearSVR(C=0.1),
            "LinearSVR, c=1": LinearSVR(C=1),
            "LinearSVR, c=10": LinearSVR(C=10),
        }
    },
    "knn": {
        'default': {
            "KNN, weights=uniform, leaf_size=30, neighbors=5": KNeighborsRegressor(weights='uniform', leaf_size=30, n_jobs=-2),
        },
        'uniform': {
            "KNN, weights=uniform, leaf_size=10, neighbors=5": KNeighborsRegressor(weights='uniform', leaf_size=10, n_jobs=-2),
            "KNN, weights=uniform, leaf_size=30, neighbors=5": KNeighborsRegressor(weights='uniform', leaf_size=30, n_jobs=-2),
            "KNN, weights=uniform, leaf_size=60, neighbors=5": KNeighborsRegressor(weights='uniform', leaf_size=60, n_jobs=-2),
        },
        'distance': {
            "KNN, weights=distance, leaf_size=10, neighbors=5": KNeighborsRegressor(weights='distance', leaf_size=10),
            "KNN, weights=distance, leaf_size=30, neighbors=5": KNeighborsRegressor(weights='distance', leaf_size=30),
            "KNN, weights=distance, leaf_size=60, neighbors=5": KNeighborsRegressor(weights='distance', leaf_size=60),
        }
    }   
}

to_be_tested_models = {}

for model_desc in models[args[0]][args[1]]:
    to_be_tested_models[model_desc] = models[args[0]][args[1]][model_desc]
    

for model_name in to_be_tested_models:
    model = to_be_tested_models[model_name]
    similarity_model = train_and_save_model(model_name, model, train_X, train_y)
    correctly_predicted_count = test_model(model_name, test_X, test_y, similarity_model)



exit()

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
