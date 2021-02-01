#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os
import time
import numpy as np
import tensorflow as tf
# from slices_select_config import config,getLabelMatrix
from slices_select_function import getLabelMatrix
from slices_select_FS import getACC,getACC1,select_slices_GA_FS,select_slices_Rank_FS
# from slices_select_FS_1 import select_slices_GA_FS
from sklearn.metrics import matthews_corrcoef
# import cnnmodel_20180927 as cnnmodel

def main():
    cv_flag = 0
    time_start = time.time()
    classify_flag = 2
    cv = 'cv'+str(cv_flag)
    Data_flag = 'Data'+str(cv_flag)
    openfile = open('MCIcvsMCInc_'+cv+'.txt','a')
    openfile.write("==================================================================================================================\n")
    openfile.write("para setting: popsize = 50,pc = 0.75,pm = 0.05,iters = 5000,chromlength = 35 \n")

    # slices_pos_list = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68,
    #                70,72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98,
    #                24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72,
    #                74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102,
    #                104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54,
    #                56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84,
    #                86, 88, 90, 92, 94]
    # slices_pos_list_xyz = ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X',
    #                        'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X',
    #                        'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X',
    #                        'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y',
    #                        'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y',
    #                        'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y',
    #                        'Y',
    #                        'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z',
    #                        'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z']
    #########################################    AD_HC   ############################
    # slices_pos_list = [34, 32, 82, 36, 76,38,74,74,72,76,70,80,78,48,62,68,72,60,42,80,66,46,44,40,84,64,84,
    #                    30,82,78,46,50,68,44,38]#,40,36,86,92,54,70,58,42,90,96,48,88,56,58,86,34,56,64,88,62,50]
    # #                    # 52,54,94,90,26,96,66,58,24,100,64,92,28,56,98,32,102,60,54,52,52,20,98,22,94,50,30,60,
    # #                    # 104,72,66,48,114,62,38,32,34,46,110,80,42,70,108,94,112,44,122,116,74,82,106,78,68,30,
    # #                    # 76,92,86,36,28,40,84,24,26,88,120,90,118]
    # slices_pos_list_xyz = ['Z','Z','Y','Z','X','Z',	'X','Y'	,'X','Y','Y','Y','Y','X','Y','Y','Y','Y','Z','X',
    #                        'Y','X','Z','Z','X','Y','Y','Z','X','X','Z','X','X','X','X']#,'X','X','Y','Y','Z',
    #                        # 'X','Z','X','Y','Y','Z','X','Z','X','X','X','X','X','Y','X','Z']
    # #                        # 'X','X','Y','X',
    #                        # 'X','X','X','Y','X','Y','Z','X','X','Y','Y','X','Y','X','Y','Z','Y','X','X','X',
    #                        # 'X','Y','X','Z','Y','Z','Z','Y','Y','Z','Y','Y','Y','Y','Y','Z','Y','Z','Y','Z',
    #                        # 'Y','Y','Y','Y','Z','Z','Y','Z','Z','Y','Z','Z','Z','Y','Y','Y','Z','Y','Y','Z',
    #                        # 'Y','Z','Y']
    # slices_pos_list = [38, 72, 76, 32, 80, 44, 34, 30, 36, 74, 70, 78, 68, 48, 82, 78, 38, 46, 30, 64, 50, 46,
    #                    40, 76, 82, 94, 74, 80, 40, 22, 62, 88, 48, 42, 28, 52, 26, 56, 98, 20, 58, 44, 50, 54, 36, 54, 48, 52, 32, 36,
    #                    46, 84, 96, 60, 92, 68, 42, 94, 66, 24, 72, 64, 34, 40, 86, 62, 68, 84, 108, 58]
    #
    # slices_pos_list_xyz = ['Z', 'X', 'Y', 'Z', 'Y', 'Z', 'Z', 'Z', 'Z', 'X', 'X', 'X', 'Z', 'X', 'X', 'Y', 'X', 'X', 'X',
    #                        'Y', 'Z', 'Z', 'X', 'X', 'Y', 'Y', 'Y', 'X', 'Z', 'X', 'Y', 'X', 'Y', 'Z', 'X', 'X', 'X', 'Y', 'X', 'X', 'Y', 'X', 'Y', 'Y',
    #                        'X', 'X', 'Z', 'Z', 'X', 'Y', 'Y', 'Y', 'X', 'Y', 'Y', 'Y', 'X', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'Y', 'X', 'X', 'X', 'X', 'Y', 'Z']
    ####################################MCIc vs MCInc##################################
    slices_pos_list = [76, 30, 34, 36, 90, 86, 50,32, 30, 84, 42, 32, 62, 86, 100, 32, 92,
                        78, 56, 66, 88, 40, 48, 38, 24, 28, 44, 80, 102, 36, 60, 82, 72,
                        26, 52, 52, 84, 54, 118, 46, 66, 68, 74, 78, 98, 20, 80, 90, 38,
                        52, 62, 92, 36, 56, 34, 42, 54, 84, 64, 22]
    slices_pos_list_xyz = ['Y', 'X', 'Z', 'Z', 'Y', 'Z', 'X', 'Z', 'Z', 'Y', 'Z', 'Y', 'Y', 'Y', 'Y',
                            'X', 'Y', 'Y', 'Y', 'Z', 'Y', 'Z', 'Z', 'Z', 'X', 'X', 'Z', 'Y', 'Y', 'Y',
                            'Y', 'Y', 'Z', 'X', 'X', 'Z', 'X', 'X', 'Y', 'X', 'Y', 'Y', 'Z', 'Z', 'Y',
                            'X', 'X', 'X', 'X', 'Y', 'Z', 'X', 'X', 'Z', 'Y', 'Y', 'Y', 'Z', 'Z', 'X']

    X_slices = []
    Y_slices = []
    Z_slices = []
    ACC_list = []
    RECALL_list = []
    PRECISION_list = []
    AUC_list = []
    MCC_list = []
    slices_pos_list = np.array(slices_pos_list)
    slices_pos_list_xyz = np.array(slices_pos_list_xyz)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=
    # fold = 0
    # result_list, real = getLabelMatrix(classify_flag, fold, slices_pos_list, slices_pos_list_xyz, 0)
    # slices_pos_selcted_index, best_fitness = select_slices_GA_FS(result_list, real)
    # print("length:%d " % len(slices_pos_selcted_index))
    # best_slices = slices_pos_list[slices_pos_selcted_index]
    # best_slices_axis = slices_pos_list_xyz[slices_pos_selcted_index]
    # x_slices_selected = []
    # y_slices_selected = []
    # z_slices_selected = []
    # for slices_index in slices_pos_selcted_index:
    #     if slices_pos_list_xyz[slices_index] == 'X':
    #         x_slices_selected.append(slices_pos_list[slices_index])
    #     elif slices_pos_list_xyz[slices_index] == 'Y':
    #         y_slices_selected.append(slices_pos_list[slices_index])
    #     else:
    #         z_slices_selected.append(slices_pos_list[slices_index])
    # X_slices.append(x_slices_selected)
    # Y_slices.append(y_slices_selected)
    # Z_slices.append(z_slices_selected)
    #
    # # openfile.write("X:"+str(X_slices)+"\n")
    # # openfile.write("Y:"+str(Y_slices)+"\n")
    # # openfile.write("Z:"+str(Z_slices)+"\n")
    # print("X:",X_slices)
    # print("Y:",Y_slices)
    # print("Z:",Z_slices)
    #
    # # test_result_list, te_real = getLabelMatrix(classify_flag, fold, best_slices, best_slices_axis, 1)
    # # acc_vote = getACC(test_result_list, te_real)
    # # openfile.write("ACC:%f\n" % acc_vote)
    # acc_vote, std_acc, auc, std_auc = cnnmodel.get_ACC_AUC(classify_flag,X_slices[0],Y_slices[0],Z_slices[0])
    # print("ACC:%f" % acc_vote)
    # openfile.close()
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for fold in range(0,5):
        x_slices_selected = []
        y_slices_selected = []
        z_slices_selected = []
        openfile.write("%d fold is working...\n" % fold)
        openfile.write("caculate result matrix is working...\n")
        print("%d fold is working..." % fold)
        print("caculate result matrix is working...")
        time1 = time.time()
        result_list,real = getLabelMatrix(classify_flag,fold,slices_pos_list,slices_pos_list_xyz, 0, Data_flag, cv)
        time2 = time.time()

        openfile.write("caculate result matrix is excuting %f min\n" % ((time2-time1)/60))
        print("caculate result matrix is excuting %f min" % ((time2-time1)/60))

        openfile.write("select best slices is working...\n")
        print("select best slices is working...")
        slices_pos_selcted_index, best_fitness= select_slices_GA_FS(result_list,real)
        # slices_pos_selcted_index, best_fitness = select_slices_Rank_FS(result_list, real)
        openfile.write("best_fitness of GA:%f " % best_fitness)
        print("best_fitness of GA:%f " % best_fitness)
        # slices_pos_selcted_index = [0,1]

        time3 = time.time()
        openfile.write("length of best slices is %d\n" % len(slices_pos_selcted_index))
        openfile.write("select best slices is excuting %f min\n" % ((time3-time2)/60))
        print("length of best slices is %d" % len(slices_pos_selcted_index))
        print("select best slices is excuting %f min" % ((time3-time2)/60))

        best_slices = slices_pos_list[slices_pos_selcted_index]
        best_slices_axis = slices_pos_list_xyz[slices_pos_selcted_index]
        # best_slices = [76, 74, 72, 48, 80, 82,74,76,70,80,34,32,36,38,42]
        # best_slices_axis = ['X','X','X','X','X','Y','Y','Y','Y','Y','Z','Z','Z','Z','Z']
        openfile.write("testing...\n")
        print("testing...")

        for slices_index in slices_pos_selcted_index:
            if slices_pos_list_xyz[slices_index] == 'X':
                x_slices_selected.append(slices_pos_list[slices_index])
            elif slices_pos_list_xyz[slices_index] == 'Y':
                y_slices_selected.append(slices_pos_list[slices_index])
            else:
                z_slices_selected.append(slices_pos_list[slices_index])
        X_slices.append(x_slices_selected)
        Y_slices.append(y_slices_selected)
        Z_slices.append(z_slices_selected)

        print("X:", x_slices_selected)
        print("Y:", y_slices_selected)
        print("Z:", z_slices_selected)

        test_result_list, te_real = getLabelMatrix(classify_flag,fold,best_slices,best_slices_axis,1, Data_flag,cv)
        time4 = time.time()
        # openfile.write("testing is excuting %f min\n" % ((time4-time3)/60))
        # print("testing is excuting %f min" % ((time4-time3)/60))

        # acc_vote1, std_acc, auc, std_auc = cnnmodel.get_ACC_AUC(classify_flag, X_slices[0], Y_slices[0], Z_slices[0])
        # print("acc_vote1:%f" % acc_vote1)
        acc_vote,recall_vote, precision_vote, auc_vote, mcc_vote = getACC(test_result_list,te_real)
        # acc_vote = getACC1(test_result_list,te_real,best_slices,best_slices_axis)
        ACC_list.append(acc_vote)
        RECALL_list.append(recall_vote)
        PRECISION_list.append(precision_vote)
        AUC_list.append(auc_vote)
        MCC_list.append(mcc_vote)
        openfile.write("the %d fold ACC is %f \n\n" % (fold,acc_vote))
        print("the %d fold ACC is %f " % (fold,acc_vote))
        openfile.write("the %d fold RECALL is %f \n\n" % (fold, recall_vote))
        print("the %d fold RECALL is %f " % (fold, recall_vote))
        openfile.write("the %d fold PRECISION is %f \n\n" % (fold, precision_vote))
        print("the %d fold PRECISION is %f " % (fold, precision_vote))
        openfile.write("the %d fold AUC is %f \n\n" % (fold, auc_vote))
        print("the %d fold AUC is %f " % (fold, auc_vote))
        openfile.write("the %d fold MCC is %f \n\n" % (fold, mcc_vote))
        print("the %d fold MCC is %f \n" % (fold, mcc_vote))
    ACC_mean = np.mean(ACC_list); RECALL_mean = np.mean(RECALL_list);PRECISION_mean = np.mean(PRECISION_list);MCC_mean = np.mean(MCC_list)
    ACC_std = np.std(ACC_list); RECALL_std = np.std(RECALL_list); PRECISION_std = np.std(PRECISION_list);MCC_std = np.std(MCC_list)
    AUC_mean = np.mean(AUC_list)
    AUC_std = np.std(AUC_list)
    openfile.write("ACC:%f +- %f\n" % (ACC_mean, ACC_std));
    print("ACC:%f +- %f" % (ACC_mean, ACC_std))
    openfile.write("RECALL:%f +- %f\n" % (RECALL_mean, RECALL_std));
    print("RECALL:%f +- %f" % (RECALL_mean, RECALL_std))
    openfile.write("PRECISION:%f +- %f\n" % (PRECISION_mean, PRECISION_std));
    print("PRECISION:%f +- %f" % (PRECISION_mean, PRECISION_std))
    openfile.write("AUC:%f +- %f\n" % (AUC_mean, AUC_std));
    print("AUC:%f +- %f" % (AUC_mean, AUC_std))
    openfile.write("MCC:%f +- %f\n" % (MCC_mean, MCC_std));
    print("MCC:%f +- %f" % (MCC_mean, MCC_std))

    openfile.write("X:"+str(X_slices)+"\n")
    openfile.write("Y:"+str(Y_slices)+"\n")
    openfile.write("Z:"+str(Z_slices)+"\n")
    print("X:",X_slices)
    print("Y:",Y_slices)
    print("Z:",Z_slices)
    time_end = time.time()
    openfile.write("The whole program is excuting %f min\n\n\n" % ((time_end - time_start)/60))
    print("The whole program is excuting %f min" % ((time_end - time_start)/60))
    openfile.close()



if __name__ == '__main__':
    main()