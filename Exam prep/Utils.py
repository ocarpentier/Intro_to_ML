import math as m
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from toolbox_02450 import mcnemar
from toolbox_02450 import similarity
def get_var_expl(S):
    tot = np.sum(S**2)
    for i,val in enumerate(S):
        print(f'By PC{i+1} {np.sum(S[0:i+1]**2)/tot*100}% is used.')
        
#prob not correct
def cost_ridgereg_1d(X,Y,w,w0,lamb):
    X_hat = (X-np.mean(X))/np.std(X)
    Y_hat = (Y-np.mean(Y))
    cost = Y_hat-(X_hat*w)-w0
    cost = np.sum(cost**2)-lamb*w**2
    return cost

def McNemar(n11,n12,n21,n22,alpha=0.05):
    nn = np.zeros((2, 2))

    nn[0, 0] = n11
    nn[0, 1] = n12
    nn[1, 0] = n21
    nn[1, 1] = n22

    n = sum(nn.flat);

    thetahat = (n12 - n21) / n
    Etheta = thetahat

    Q = n ** 2 * (n + 1) * (Etheta + 1) * (1 - Etheta) / ((n * (n12 + n21) - (n12 - n21) ** 2))

    p = (Etheta + 1) * 0.5 * (Q - 1)
    q = (1 - Etheta) * 0.5 * (Q - 1)

    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1 - alpha, a=p, b=q))

    p = 2 * scipy.stats.binom.cdf(min([n12, n21]), n=n12 + n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12 + n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=", (n12 + n21))

    print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    return thetahat, CI, p

def Jeffreys(m, n, alpha=0.05):
    """
    :param m: number of accurate guesses
    :param n: number of total geusses
    :param alpha:
    :return:
    """
    # m = sum(y - yhat == 0)
    # n = y.size
    a = m+.5
    b = n-m + .5
    CI = scipy.stats.beta.interval(1-alpha, a=a, b=b)
    thetahat = a/(a+b)
    print(f'a is {a} and b is {b}')
    return thetahat, CI

def Impurity(Parent,childrens,method):
    """"
    Parent: List of number of objects belonging to corresponding classes
    childrens: List containing a list for every branch with number of objects belonging to classes
    method: A function to calculate the impurity
    """

    epsilon = 0
    for child in childrens:
        # I_ch.append(method(child))
        I_ch = method(child)
        epsilon += sum(child)/sum(Parent)*I_ch
    impurity_gain = method(Parent) - epsilon
    return impurity_gain

def Classerror(C_V):
    prob = []
    for idx,val in enumerate(C_V):
        prob.append(val/sum(C_V))
    return 1-max(prob)

def Entropy(C_V):
    entrop = 0
    for c in C_V:
        entrop -= c/sum(C_V)*m.log2(c/sum(C_V))
    return entrop

def Gini(C_V):
    gin = 0
    for c in C_V:
        gin += (c/sum(C_V))**2
    return 1-gin

def PCA_variances(S_arr):
    som = np.sum(S_arr**2)
    for idx,S in enumerate(S_arr):
        print(f'PCA component number {idx+1} explains {S**2/som*100}% of the variance')
        print(f'PCA components    0- {idx+1} explains {np.sum(S_arr[0:idx+1]**2)/som*100}%')
        print('\n')

def sigmoid(X):
    A = 1 / (1 + (np.exp((-X))))
    return A

def softmax(x):
    return  np.exp(x)/sum(np.exp(x)),1/sum(np.exp(x))

def ROC(TP,TN,FP,FN):
    print(f'TPR = {TP/(FN+TP)}')
    print(f'FPR = {FP/(FP+TN)}')


def normal_dens(mu,sigma,x):
    return 1/m.sqrt(2*m.pi*sigma**2)*m.exp(-0.5*((x-mu)/sigma)**2)

def ARD(distances):
    """
    :param distances: List of K nearest neighboures' starting with the searched one
    :return: average relative distance
    """
    densities = []
    K = len(distances[0])
    for dist in distances:
        densities.append(1/(1/K*sum(dist)))

    ard = densities[0]/(1/K*sum(densities[1::]))
    return ard

def confusion_mat_info(TP,TN,FN,FP):
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)

    ROC(TP,TN,FP,FN)

    print(f'Precion is {Precision}')
    print(f'Recall is {Recall}')
    print(f'using F-measure: F1 = {2*Precision*Recall/(Precision+Recall)}')

def Jaccard_index(pred,target):
    """
    jaccard index for a two class 1d targets.
    :param pred: 1d numpy arraywith the predicted class labels
    :param target: 1d numpy arraywith the target class labels
    :return: The jaccard index
    """
    if (type(pred) != type(np.ones(10))) or  (type(target) != type(np.ones(10))):
        raise Exception('Inputs should be 1D numpy arrays')

    som = pred+target
    num_zeros = (som == 0).sum()
    num_ones = (som == 1).sum()
    num_twos = (som == 2).sum()

    S = num_zeros*(num_zeros-1)/2+num_twos*(num_twos-1)/2
    D = num_twos*num_zeros
    N = num_zeros+num_ones+num_twos
    J = S/(N/2*(N-1)-D)
    return J


if __name__ == '__main__':
    # # get_var_expl(np.array([43.67,33.47,31.15,30.36,27.77,13.86]))
    # X = np.array([2,5,6,7])
    # Y = np.array([6,7,7,9])
    # # print(cost_ridgereg_1d(X, Y, 0.6, 0, 2))
    # # print(McNemar(134,40,24,47)[0])
    # m = 134+141+131+132+24+26+50
    # n = m+40+31+23+30+47+48+66+58
    # print(Jeffreys(m=m,n=n))
    #
    # Parent = [5,10]
    # childs = [[1,4],[4,6]]
    # print(Impurity(Parent,childs,Gini))
    #
    # sigma = np.array([19.64,6.87,3.26,2.3,1.12])
    # PCA_variances(sigma)

    parent = [8,3]
    child  = [[6,1],[2,2]]
    print(Impurity(parent,child,Gini))
    # X = np.mat([1,3,3])
    # W1 = np.mat([[-1.2,-1],[-1.3,0],[0.6,0.9]])
    # W2 = np.mat([[-0.3,0.5]])
    #
    # print(sigmoid(X*W1)*W2.transpose()+2.2)
    # W1 = np.mat([-0.77,-5.54,0.01])
    # W2 = np.mat([0.26,-2.09,-0.03])
    # b1 = 0.0
    # b2 = 0
    # y1 = np.mat([1,b1,b2])*W1.transpose()
    # y2 = np.mat([1,b1,b2])*W2.transpose()
    # print(softmax(np.concatenate((y1,y2))))

    # o1 = [0,0,0,0,0,0,0,0,0]
    # o2 = [0,0,0,0,0,0,0,0,1]
    # o3 = [0,1,1,1,1,1,0,0,0]
    # o4 = [1,0,0,0,0,0,0,0,0]
    # print(similarity(o2,o4,'smc'))

    # print(ROC(3,1,1,2))
    # dist = [[0.9,1.],[1,1.3],[0.9,1.3]]
    # print(ARD(dist))

    # print(softmax(np.array([1.41 + 0.76*(-0.06) + 1.76*(-0.28) -0.32*0.43 -0.96*(-0.3) + 6.64*(-0.36) -2.74])))
    # confusion_mat_info(34,39,7,11)

    pred = np.array([1,1,1,1,1,1,1,1,0,0,1])
    tar  = np.array([1,1,1,1,1,1,1,1,0,0,0])
    print(Jaccard_index(pred,tar))