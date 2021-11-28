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
import os
import random
import re
import pickle

import numpy as np
import pandas as pd
#from Bio import Align
#from transformers import AutoModel, AutoTokenizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from lightgbm import LGBMRegressor

import optuna

import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split


logging.basicConfig(filename="LOGS.txt",
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


def get_protbert_embedding(aa_sequence: str):
    cleaned_sequence = re.sub(r'[UZOB]', 'X', aa_sequence)
    #tokens = PROTEIN_TOKENIZER(cleaned_sequence, return_tensors='pt')
    #output = PROTBERT(**tokens)
    #return output.last_hidden_state.detach().numpy().mean(axis=1)
    return ''

def split_data(similarity_df, protein_sequences_vectorized,  train_data_size: int):
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


def train_protein_similarity_model_SVR(train_X, train_y):
    clf = {}
    
    clf = svm.LinearSVR(max_iter=1000)
    
    logging.info('Training model')
    clf.fit(np.array(train_X), np.array(train_y))
    logging.info('Training completed.')

    with open('linear_svr.pickle', 'wb') as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('Saved the model as a pickle.\n')

    return clf

def train_protein_similarity_model_LGBM(train_X, train_Y):
    model = LGBMRegressor()

    logging.info('Training model')
    model.fit(np.array(train_X), np.array(train_Y))

    # for cross validation
    #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    #n_scores = cross_val_score(model, np.array(train_X), np.array(train_Y), scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    #print(n_scores)
    #print('MAE: %.3f (%.3f)' % (np.mean(n_scores),np.std(n_scores)))


    logging.info('Training completed.')

    with open('LGBM.pickle', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('Saved the model as a pickle.\n')

    return model

def test_model(test_X, test_y, similarity_model):
    c = 0
    for vector, actual in zip(test_X, test_y):
        prediction = similarity_model.predict(np.array(vector).reshape(1, -1))

        #print(f'Prediction: {prediction}, Actual: {actual}, difference: {abs(prediction - actual)}')
        if abs(prediction - actual) <= 0.001:
            c += 1

    print(f'{c} out of {len(test_X)} samples are predicted close to correct.')
    logging.info(f'{c} out of {len(test_X)} samples are predicted close to correct.')
    print(f'Accuracy: {float(c) / len(test_X)}')
    logging.info(f'Accuracy: {float(c) / len(test_X)}')

    return c


# --------------  OPTUNA  ------------------

# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial, data, target):
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dvalid = lgb.Dataset(valid_x, label=valid_y)

    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    gbm = lgb.train(param, dtrain, valid_sets=[dvalid], callbacks=[pruning_callback])

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy


similarity_df = get_similarity_df('sw_sim_matrix.csv')
protein_sequences_vectorized = get_protein_sequences_vectorized(
    'proteins.json', get_protbert_embedding)

train_X, train_y, test_X, test_y = split_data(similarity_df, protein_sequences_vectorized, train_data_size=450)

print(len(train_X))
print(len(train_y))
print(len(test_X))
print(len(test_y))

similarity_model = train_protein_similarity_model_SVR(train_X, train_y)
correctly_predicted_count = test_model(test_X, test_y, similarity_model)

#similarity_model = train_protein_similarity_model_LGBM(train_X, train_y)
#correctly_predicted_count = test_model(test_X, test_y, similarity_model)


exit()

study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize")
study.optimize(lambda trial: objective(trial, train_X, train_y), n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


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
