"""
Most of the code in this file is adapted from the pyCausalFS package
(https://github.com/wt-hu/pyCausalFS) by Wentao Hu et al., 2020.
"""

from itertools import combinations
from typing import Callable

import numpy as np


def HITON_PC(data, target, alaph, ci_test):
    _, kVar = np.shape(data)
    sepset = [[] for i in range(kVar)]
    variDepSet = []
    candidate_PC = []
    PC = []
    ci_number = 0
    max_k = 3

    # use a list to store variables which are not condition independence with
    # target,and sorted by dep max to min
    candidate_Vars = [i for i in range(kVar) if i != target]
    for x in candidate_Vars:
        ci_number += 1
        pval_gp = ci_test(target, x, [])
        if pval_gp <= alaph:
            variDepSet.append([x, 1 - pval_gp])

    # sorted by dep from max to min
    variDepSet = sorted(variDepSet, key=lambda x: x[1], reverse=True)
    # print(variDepSet)

    # get number by dep from max to min
    for i in range(len(variDepSet)):
        candidate_PC.append(variDepSet[i][0])
    # print(candidate_PC)

    """ sp """
    for x in candidate_PC:
        PC.append(x)
        PC_index = len(PC)
        # if new x add will be removed ,test will not be continue,so break the
        # following circulate to save time ,but i don't not why other index
        # improve
        breakFlagTwo = False

        while PC_index >= 0:
            #  reverse traversal PC,and use PC_index as a pointer of PC
            PC_index -= 1
            y = PC[PC_index]
            breakFlag = False
            conditions_Set = [i for i in PC if i != y]

            if len(conditions_Set) >= max_k:
                Slength = max_k
            else:
                Slength = len(conditions_Set)

            for j in range(Slength + 1):
                SS = set(combinations(conditions_Set, j))
                for s in SS:
                    ci_number += 1
                    conditions_test_set = [i for i in s]
                    pval_rm = ci_test(target, y, conditions_test_set)
                    if pval_rm > alaph:
                        sepset[y] = [i for i in conditions_test_set]
                        # if new x add will be removed ,test will not be
                        # continue
                        if y == x:
                            breakFlagTwo = True
                        PC.remove(y)
                        breakFlag = True
                        break

                if breakFlag:
                    break
            if breakFlagTwo:
                break

    return list(set(PC)), sepset, ci_number


def HITON_MB(data, target, alaph, ci_test):

    PC, sepset, ci_number = HITON_PC(data, target, alaph, ci_test)
    # print("PC is:" + str(PC))
    currentMB = PC.copy()
    for x in PC:
        # print("x is: " + str(x))
        PCofPC, _, ci_num2 = HITON_PC(data, x, alaph, ci_test)
        ci_number += ci_num2
        # print("PCofPC is " + str(PCofPC))
        for y in PCofPC:
            # print("y is " + str(y))
            if y != target and y not in PC:
                conditions_Set = [i for i in sepset[y]]
                conditions_Set.append(x)
                conditions_Set = list(set(conditions_Set))
                ci_number += 1
                pval = ci_test(target, y, conditions_Set)
                if pval <= alaph:
                    # print("append is: " + str(y))
                    currentMB.append(y)
                    break

    return list(set(currentMB)), ci_number


def CausalSearch(T, PCT, Z, IDT, alaph, idT3, idT3_count, idT4, idT4_count, ci_test):
    num_ci = 0
    # step 1:Single PC
    if len(PCT) == 1:
        IDT[T, PCT[0]] = 3

    # step 2:Check C2 & C3
    for i in range(len(PCT)):
        for j in range(len(PCT)):
            if i != j:
                x = PCT[i]
                y = PCT[j]
                if x in Z or y in Z:
                    continue
                # print("X is: ",x," y is: ",y," Z is: ", Z)
                pval = ci_test(x, y, Z)
                num_ci += 1
                condition_vars = [i for i in Z]
                condition_vars.append(T)
                condition_vars = sorted(set(condition_vars))
                pval2 = ci_test(x, y, condition_vars)
                num_ci += 1
                if pval > alaph and pval2 <= alaph:
                    IDT[T, x] = 1
                    IDT[T, y] = 1
                elif pval <= alaph and pval2 > alaph:
                    if IDT[T, x] == 1:
                        IDT[T, y] = 2
                    elif IDT[T, y] != 2:
                        IDT[T, y] = 3
                    if IDT[T, y] == 1:
                        IDT[T, x] = 2
                    elif IDT[T, x] != 2:
                        IDT[T, x] = 3
                    # add(X,Y)to pairs with idT=3
                    idT3_count += 1
                    idT3.append([x, y])
                else:
                    if (IDT[T, x] == 0 and IDT[T, y] == 0) or (
                        IDT[T, x] == 4 and IDT[T, y] == 4
                    ):
                        IDT[T, x] = 4
                        IDT[T, y] = 4
                    # add(X,Y) to pairs with idT=4
                    idT4_count += 1
                    idT4.append([x, y])

    # step 3:identify idT=3 pairs with known parents
    for i in range(len(PCT)):
        x = PCT[i]
        if IDT[T, x] == 1:
            for j in range(idT3_count):
                if idT3[j][0] == x:
                    y = idT3[j][1]
                    IDT[T, y] = 2
                elif idT3[j][1] == x:
                    y = idT3[j][0]
                    IDT[T, y] = 2
    return IDT, idT3, idT3_count, idT4, idT4_count, num_ci


def CMB_subroutine(Data, T, alaph, IDT, already_calculated_MB, all_MB, ci_test):

    # already_calculated_MB[T] = 0
    Z = []
    idT3 = []
    idT3_count = 0
    idT4 = []
    idT4_count = 0
    num_ci = 0
    PCT, _, n_c = HITON_PC(Data, T, alaph, ci_test)
    num_ci += n_c
    IDT, idT3, idT3_count, idT4, idT4_count, n_c1 = CausalSearch(
        T, PCT, Z, IDT, alaph, idT3, idT3_count, idT4, idT4_count, ci_test
    )
    num_ci += n_c1
    # step 2:further test variables with idT=4
    for i in range(idT4_count):
        x = idT4[i][0]
        y = idT4[i][1]
        if already_calculated_MB[x] == 1:
            all_MB[x], n_c2 = HITON_MB(Data, x, alaph, ci_test)
            num_ci += n_c2

            already_calculated_MB[x] = 0
        Z = []
        if x in all_MB.keys():
            Z = [i for i in all_MB[x] if i != T and i != y]
        IDT, idT3, idT3_count, idT4, idT4_count, n_c3 = CausalSearch(
            T, PCT, Z, IDT, alaph, idT3, idT3_count, idT4, idT4_count, ci_test
        )
        num_ci += n_c3
        if 4 not in IDT:
            break
    parents = [idx for idx, i in enumerate(IDT[T]) if i == 1]
    for i in range(len(parents)):
        x = parents[i]
        for j in range(len(parents)):
            if j != i:
                y = parents[j]
                for k in range(idT4_count):
                    if idT4[k][0] == x:
                        z = idT4[k][1]
                        for l in range(idT4_count):
                            if l != k:
                                if (idT4[l][0] == y and idT4[l][1] == z) or (
                                    idT4[l][0] == z and idT4[l][1] == y
                                ):
                                    IDT[T, z] = 1
                    elif idT4[k][1] == x:
                        z = idT4[k][0]
                        for l in range(idT4_count):
                            if l != k:
                                if (idT4[l][0] == y and idT4[l][1] == z) or (
                                    idT4[l][0] == z and idT4[l][1] == y
                                ):
                                    IDT[T, z] = 1
    for idx, i in enumerate(IDT[T]):
        if i == 4:
            IDT[T, idx] = 3

    return IDT, idT3, idT3_count, PCT, num_ci


def Meek(DAG, pdag, Data):
    n, p = np.shape(Data)
    old_pdag = np.zeros((p, p))

    while not (pdag == old_pdag).all():
        old_pdag = pdag.copy()
        # rule 1 a->b-c ===>   a->b->c
        X = [i for i in range(p) for j in range(p) if pdag[i, j] == -1]
        Y = [j for i in range(p) for j in range(p) if pdag[i, j] == -1]
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            Z = [j for j in range(p) if pdag[y, j] == 1 and DAG[x, j] == 0]
            for z in Z:
                pdag[y, z] = -1
                pdag[z, y] = 0
                DAG[y, z] = 1
                DAG[z, y] = 0
                # G[y, z] = 1
                # G[z, y] = 0
        # rule 2 a->c->b,a-b ===>  a->b
        X = [i for i in range(p) for j in range(p) if pdag[i, j] == 1]
        Y = [j for i in range(p) for j in range(p) if pdag[i, j] == 1]
        if len(X) == 0:
            break
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            if np.any(
                np.multiply(np.array(pdag[x, :] == -1), np.array(pdag[:, y] == -1))
            ):
                pdag[x, y] = -1
                pdag[y, x] = 0
                DAG[x, y] = 1
                DAG[y, x] = 0
                # G[x, y] = 1
                # G[y, x] = 0

        # rule 3 a-c->b,a-d->b,a-b ===>  a->b
        X = [i for i in range(p) for j in range(p) if pdag[i, j] == 1]
        Y = [j for i in range(p) for j in range(p) if pdag[i, j] == 1]
        if len(X) == 0:
            break
        for i in range(len(X)):
            a = X[i]
            b = Y[i]
            C = [m for m in range(p) if pdag[m, b] == -1 and pdag[a, m] == 1]
            for c in C:
                for d in C:
                    if c != d and pdag[c, d] == 0 and pdag[d, c] == 0:
                        pdag[a, b] = -1
                        pdag[b, a] = 0
                        DAG[a, b] = 1
                        DAG[b, a] = 0
    return pdag


def CMB(Data, T, alaph, ci_test):
    n, p = np.shape(Data)
    DAG = np.zeros((p, p))
    pdag = np.zeros((p, p))
    G = np.zeros((p, p))
    Tmp = []
    Q = [T]
    all_idT3 = {}
    all_idT3_count = [0] * p
    already_calculated = [1] * p
    already_calculated_MB = [1] * p
    all_MB = {}
    break_flag = False
    num_ci = 0
    # Step 1:establish initial ID
    IDT = np.zeros((p, p))

    # if no element of IDT is equal to 3,break
    while len(Tmp) <= p and len(Q) != 0:
        A = Q[0]
        Q.remove(A)
        if A in Tmp:
            continue
        else:
            Tmp.append(A)
        if already_calculated[A]:
            IDT, all_idT3[A], all_idT3_count[A], pctemp, n_c = CMB_subroutine(
                Data, A, alaph, IDT, already_calculated_MB, all_MB, ci_test
            )
            num_ci += n_c
            already_calculated[A] = 0
        IDT_A_3 = [index for index, i in enumerate(IDT[A]) if i == 3]
        IDT_A_2 = [index for index, i in enumerate(IDT[A]) if i == 2]
        IDT_A_1 = [index for index, i in enumerate(IDT[A]) if i == 1]

        for i in IDT_A_3:
            DAG[A, i] = 1
            DAG[i, A] = 1
        for i in IDT_A_2:
            DAG[A, i] = 1
            DAG[i, A] = 1
        for i in IDT_A_1:
            DAG[A, i] = 1
            DAG[i, A] = 1

        for i in IDT_A_3:
            pdag[A, i] = 1
            pdag[i, A] = 1
        for i in IDT_A_2:
            pdag[A, i] = -1
            pdag[i, A] = 0
        for i in IDT_A_1:
            pdag[A, i] = 0
            pdag[i, A] = -1

        for i in IDT_A_3:
            G[A][i] = 1
            G[i][A] = 1
        for i in IDT_A_2:
            G[A][i] = 1
            G[i][A] = 0
        for i in IDT_A_1:
            G[A][i] = 0
            G[i][A] = 1

        if 1 not in pdag[T] and 1 not in pdag[:, T]:
            break

        # Step 3:resolve variable set with idT=3
        IDT3_count = [index for index, i in enumerate(IDT[A]) if i == 3]
        for z in range(len(IDT3_count)):
            X = IDT3_count[z]
            Q.append(X)
            if already_calculated[X]:
                IDT, all_idT3[X], all_idT3_count[X], pctemp, n_c2 = CMB_subroutine(
                    Data, X, alaph, IDT, already_calculated_MB, all_MB, ci_test
                )
                already_calculated[X] = 0
                num_ci += n_c2
            # update IDT according to IDX
            if IDT[X, A] == 2:
                IDT[A, X] = 1
                for j in range(all_idT3_count[X]):
                    if all_idT3[X][j][0] == X:
                        Y = all_idT3[X][j][1]
                        IDT[A, Y] = 2
                    elif all_idT3[X][j][1] == X:
                        Y = all_idT3[X][j][0]
                        IDT[A, Y] = 2
            IDT_X_3 = [index for index, i in enumerate(IDT[X]) if i == 3]
            IDT_X_2 = [index for index, i in enumerate(IDT[X]) if i == 2]
            IDT_X_1 = [index for index, i in enumerate(IDT[X]) if i == 1]

            for i in IDT_X_3:
                DAG[X, i] = 1
                DAG[i, X] = 1
            for i in IDT_X_2:
                DAG[X, i] = 1
                DAG[i, X] = 1
            for i in IDT_X_1:
                DAG[X, i] = 1
                DAG[i, X] = 1

            for i in IDT_X_3:
                pdag[X, i] = 1
                pdag[i, X] = 1
            for i in IDT_X_2:
                pdag[X, i] = -1
                pdag[i, X] = 0
            for i in IDT_X_1:
                pdag[X, i] = 0
                pdag[i, X] = -1

            for i in IDT_X_3:
                G[X, i] = 1
                G[i, X] = 1
            for i in IDT_X_2:
                G[X, i] = 1
                G[i, X] = 0
            for i in IDT_X_1:
                G[X, i] = 0
                G[i, X] = 1

            pdag = Meek(DAG, pdag, Data)
            if 1 not in pdag[T] and 1 not in pdag[:, T]:
                break_flag = 1
                break
        if break_flag:
            break

    parents = [i for i in range(p) if pdag[i, T] == -1]
    children = [j for j in range(p) if pdag[T, j] == -1]
    undirected = [i for i in range(p) if pdag[T, i] == 1]
    PC = list(set(parents).union(set(children)).union(set(undirected)))
    return parents, children, PC, undirected, num_ci


def cmb_alg(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    target: int,
    *wargs,
    **kwargs,
):
    parents, children, _, undirected, _ = CMB(data, target, alpha, ci_test)

    g = np.zeros((data.shape[1], data.shape[1]), dtype=int)
    # Orient parents
    g[parents, target] = -1
    g[target, parents] = 1
    # Orient children
    g[target, children] = -1
    g[children, target] = 1
    # Orient undirected edges
    g[undirected, target] = -1
    g[target, undirected] = -1
    return g
