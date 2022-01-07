import pandas as pd


def Average(lst):
    return sum(lst) / len(lst)

df = pd.read_csv('sw_sim_matrix.csv')
scores1 = []
scores2 = []
scores3 = []
scores4 = []
scores5 = []
scores6 = []
scores7 = []
scores8 = []
scores9 = []
scores10 = []
scores = []
c = 0
for _, score_list in df.iterrows():
    for score in score_list[1:]:
        if score < 0.1:
            scores1.append(score)
        elif score < 0.2:
            scores2.append(score)
        elif score < 0.3:
            scores3.append(score)
        elif score < 0.4:
            scores4.append(score)
        elif score < 0.5:
            scores5.append(score)
        elif score < 0.6:
            scores6.append(score)
        elif score < 0.7:
            scores7.append(score)
        elif score < 0.8:
            scores8.append(score)
        elif score < 0.9:
            scores9.append(score)
        elif score < 1:
            scores10.append(score)
        else:
            scores.append(score)

print('<0.1', len(scores1))
print('<0.2', len(scores2))
print('<0.3', len(scores3))
print('<0.4', len(scores4))
print('<0.5', len(scores5))
print('<0.6', len(scores6))
print('<0.7', len(scores7))
print('<0.8', len(scores8))
print('<0.9', len(scores9))
print('<1', len(scores10))
print('=1', len(scores))
