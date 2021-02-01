#!/usr/bin/env python
#-*- coding: utf-8 -*-
import copy
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,precision_score
from cnnmodel_hy import concat_path,CNN_classifier,test_on_best_models
from sklearn.metrics import matthews_corrcoef

class FS_conf:
    def __init__(self):
        self.popsize = 50
        self.chromlength = 35
        self.pc = 0.75
        self.pm = 0.05
        self.iters = 5000


def initpop(popsize, chromlength):
    # pop = np.random.randint(0, 2, (popsize, chromlength))
    pop = np.random.choice(2,size=(popsize,chromlength),replace=True,p=[0.9,0.1])
    print("pop1",np.sum(pop[1]))
    print("pop2",np.sum(pop[2]))
    print("pop3",np.sum(pop[3]))
    return pop

def OneZero2Features(index, pop,slices_pos_list):
    slices_pos_list=np.array(slices_pos_list)
    dim_index = (pop[index, :] == 1)*1
    x_index = np.where(dim_index == 1)
    x_index = np.array(x_index)
    slices_pos = slices_pos_list[x_index[0]]
    return slices_pos

def fitness_Stretch(fitness):
    fitness_sum = 0
    for i in fitness:
        fitness_sum+=np.exp(i)
    for j in range(len(fitness)):
        fitness[j] = np.exp(fitness[j])/fitness_sum
    return fitness

def cal_objvalue(pop, slices_pos_list,result_matrix,te_real):
    px, py = np.shape(pop)
    objvalue = np.zeros((px, 1))
    for i in range(px):
        te_pred = []
        slices_pos_selected= OneZero2Features(i,pop,slices_pos_list)
        if len(slices_pos_selected) == 0:
            continue
        for p in slices_pos_selected:
            te_pred.append(result_matrix[p])
            # 根据预测标签投票，对三个模型的预测标签求和
        threshold = len(slices_pos_selected) / 2

        pred_vote = np.sum(te_pred, axis=0)

        pred_vote[pred_vote < threshold] = 0
        pred_vote[pred_vote >= threshold] = 1

        acc_vote = accuracy_score(te_real, pred_vote)
        # if acc_vote>0.82:
        #     objvalue[i] = 0.9*acc_vote+0.1*(1/len(slices_pos_selected))
        # else:
        #     objvalue[i] = acc_vote
        # objvalue[i] = 0.8 * acc_vote + 0.2 * (1 / len(slices_pos_selected))
        objvalue[i] = acc_vote
    # objvalue = objvalue*10
    # objvalue = fitness_Stretch(objvalue)
    return objvalue

def selection(pop, fitvalue):
    px, py = np.shape(pop)
    totalfit = np.sum(fitvalue)
    p_fitvalue = fitvalue/totalfit
    p_fitvalue = np.cumsum(p_fitvalue)
    # newpop = []
    newpop = np.zeros((np.shape(pop)))
    ms = np.sort(np.random.rand(px))
    fitin = 0
    newin = 0
    while newin < px:
        if ms[newin] < p_fitvalue[fitin]:
            # newpop.append(pop[fitin])
            newpop[newin, :] = copy.copy(pop[fitin])
            newin = newin + 1
        else:
            fitin = fitin + 1
    return newpop


def crossover(pop, pc):
    px, py = np.shape(pop)
    newpop = np.ones((px, py))
    # print(np.append(pop[3, 0:5],pop[4, 6:py]))
    for i in range(0, px-1, 2):
        if (np.random.rand() < pc):
            cpoint = int(np.random.rand() * py)
            # newpop[i, :] = pop[i,0:cpoint].extend(pop[i+1,cpoint+1:py])
            # newpop[i+1,:] = pop[i+1,0:cpoint].extend(pop[i,cpoint+1:py])
            newpop[i, :] = copy.copy(np.append(pop[i, 0:cpoint+1], pop[i+1, cpoint+1:py]))
            newpop[i+1, :] = copy.copy(np.append(pop[i+1, 0:cpoint+1], pop[i, cpoint+1:py]))
        else:
            newpop[i, :] = copy.copy(pop[i, :])
            newpop[i+1, :] = copy.copy(pop[i+1, :])
    return newpop


def mutation(pop, pm):
    px, py = np.shape(pop)
    newpop = np.ones((px, py))
    for i in range(px):
        newpop[i,:] = copy.copy(pop[i, :])
        mpoint = int(np.random.rand() * py)
        if np.random.rand() < pm:
            if newpop[i, mpoint] == 0:
                newpop[i,mpoint] = 1
            elif newpop[i, mpoint] == 1:
                newpop[i, mpoint] = 0
            else:
                newpop[i, :] = pop[i, :]
    return newpop

def best(pop, fitvalue):
    px, py = np.shape(pop)
    bestindividual = copy.copy(pop[0,:])
    bestfit = copy.copy(fitvalue[0])
    bestIndex = 0
    for i in range(1,px):
        if fitvalue[i] > bestfit:
            bestindividual = copy.copy(pop[i,:])
            bestfit = copy.copy(fitvalue[i])
            bestIndex = i
    return bestindividual, bestfit, bestIndex


def select_slices_GA_FS(result_matrix,real):
    conf = FS_conf()
    slices_pos_list = list(range(conf.chromlength))
    pop = initpop(conf.popsize, conf.chromlength)
    best_acc = 0
    index_sum = 0
    individual_index=[]

    for i in range(conf.iters):
        objvalue = cal_objvalue(pop, slices_pos_list,result_matrix,real)
        fitvalue = objvalue
        newpop_selct = selection(pop, fitvalue)
        newpop_cross = crossover(newpop_selct, conf.pc)
        newpop_mutate = mutation(newpop_cross, conf.pm)
        pop = newpop_mutate
        fitvalue_current = cal_objvalue(newpop_mutate,slices_pos_list,result_matrix,real)

        bestindividual, bestfit, bestIndex = best(newpop_mutate, fitvalue_current)
        if bestfit > best_acc:
            best_acc = copy.copy(bestfit)
            individual_index = 1 * (bestindividual == 1)
            index_sum = np.sum(individual_index)
    print("best_fitness of GA:%f " % best_acc)
    individual_index_selected = np.where(individual_index==1)
    individual_index_selected = np.array(individual_index_selected)
    slices_pos_list = np.array(slices_pos_list)
    slices_pos_selcted = slices_pos_list[individual_index_selected[0]]
    return np.array(slices_pos_selcted),best_acc


def getACC(test_result_list,te_real):
    threshold = len(test_result_list) / 2
    pred_vote = np.sum(test_result_list, axis=0)
    pred_vote[pred_vote < threshold] = 0
    pred_vote[pred_vote >= threshold] = 1

    acc_vote = accuracy_score(te_real, pred_vote)
    recall_vote = recall_score(te_real,pred_vote)
    precision_vote = precision_score(te_real,pred_vote)
    auc_vote = roc_auc_score(te_real,pred_vote)
    mcc_vote = matthews_corrcoef(te_real, pred_vote)
    return acc_vote,recall_vote,precision_vote,auc_vote, mcc_vote

def getACC1(test_result_list,te_real,slices_pos,slices_pos_xyz):
    # test_result_list = np.array(test_result_list)
    global_pred_labels = []

    te_pred_x = []
    te_pred_y = []
    te_pred_z = []
    te_pred = []
    for z in range(len(slices_pos)):
        if slices_pos_xyz[z]== 'X':
            te_pred_x.append(test_result_list[z])
        elif slices_pos_xyz[z] == 'Y':
            te_pred_y.append(test_result_list[z])
        else:
            te_pred_z.append(test_result_list[z])
    te_pred = [te_pred_x,te_pred_y,te_pred_z]
    # te_pred.append(te_pred_x);te_pred.append(te_pred_y);te_pred.append(te_pred_z)

    # 根据预测标签投票，对三个模型的预测标签求和
    threshold = len(te_pred) / 2

    pred_vote = np.sum(te_pred, axis=0)
    pred_vote[pred_vote < threshold] = 0
    pred_vote[pred_vote >= threshold] = 1


    acc_vote = accuracy_score(te_real, pred_vote)
    return acc_vote

def select_slices_Rank_FS(result_matrix,real):
    conf = FS_conf()
    slices_pos_list = list(range(conf.chromlength))
    individual_index = []
    best_acc = 0
    te_pred = []
    slices_index = []
    slices_index_ex = []
    for i in range(len(slices_pos_list)-1):
        slices_index_ex.append(i)
        for p in slices_index:
            te_pred.append(result_matrix[p])
        # 根据预测标签投票，对三个模型的预测标签求和
        threshold = len(slices_index) / 2
        # if len(slices_index) == 1:
        #     pred_vote = te_pred
        # else:
        #     pred_vote = np.sum(te_pred, axis=0)
        pred_vote = np.sum(te_pred, axis=0)
        pred_vote[pred_vote < threshold] = 0
        pred_vote[pred_vote >= threshold] = 1

        acc_vote = accuracy_score(real, pred_vote)
        print(":::::::::::::::acc_vote:%f " % acc_vote)
        if acc_vote>best_acc:
            best_acc = copy.copy(acc_vote)
            individual_index = copy.copy(slices_index)
    return individual_index,best_acc

