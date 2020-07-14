import numpy as np
import sys,os
from evaluation import nDCG5, save_result 
import data
import json

if __name__ == '__main__':
    score= []
    num  = len(sys.argv) -1
    for i in range(num):
        score_path = sys.argv[i+1]
        score.append(np.load(score_path, allow_pickle=True).item())

    score_enm = {} 
    for k in score[0].keys():
        score_list = []
        for i in range(len(score[0][k])):
            score_list.append((0, score[0][k][i][1]))
        score_enm[k] = score_list

    for k in score_enm.keys():
        for ind in range(num):
            for i in range(len(score_enm[k])):
                for j in range(len(score[ind][k])):
                    if score_enm[k][i][1] == score[ind][k][j][1]:
                        if ind ==0:
                            score_enm[k][i] = (score_enm[k][i][0] + (1./6.)*score[ind][k][j][0], score_enm[k][i][1])
                        elif ind >= 1 and ind <= 5:
                            score_enm[k][i] = (score_enm[k][i][0] +(1./6.)* score[ind][k][j][0], score_enm[k][i][1])
                        break
        #score_enm[k].sort(key=lambda x: x[0], reverse=True)
        score_enm[k].sort(key=lambda x: x[0], reverse=False)
    f = open('../data/valid/valid_answer.json')
    #answer = json.load(f)
    answer = None 
    if answer is None:
        save_result(score_enm)
    else:
        ndcg5 = nDCG5(score_enm, answer)
        print('Text to image nDCG5: ', ndcg5)

