from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from utils_optimization import optCE, optSAGD, optSCIPY, optSCIPY_bivar, optSCIPY_bivar_bis
from scipy.special import softmax
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score # weighted
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from synthetic_data import make_unfair_poolclassif, data_viz_tsne
import time
import seaborn as sns
from fairlearn.reductions import ExponentiatedGradient, DemographicParity


#------------------------------------------------------------#
# utils for multi-class fairness
#------------------------------------------------------------#

def get_multiclass_performance(model, X_train, X_test, y_train, y_test, print_results = True):

    # define the ovr strategy
    ovr = OneVsRestClassifier(model)

    # fit model
    ovr.fit(X_train[:, :-1], y_train)

    # make predictions
    yhat_proba = ovr.predict_proba(X_test[:, :-1]) # not a softmax, just a normalization on the positives probs after OvR
    yhat = ovr.predict(X_test[:, :-1])

    print("test :", accuracy_score(y_test, ovr.predict(X_test[:, :-1])))
    print("train :", accuracy_score(y_train, ovr.predict(X_train[:, :-1])))

    C = confusion_matrix(y_test, yhat)
    plot_confusion_matrix(C)
    
    return score_test, score_train


def print_fairness_results(acc_dict, ks_dict, time_dict = None, model_name = ""):
    """
    print fairness results
    """
    print("Accuracies")
    for name in acc_dict.keys():
        print(f'{model_name} {name} : {round(np.mean(acc_dict[name]), 2)} +-{round(np.std(acc_dict[name]), 3)}')
    print()
    print("Unfairness")
    for name in ks_dict.keys():
        print(f'{model_name} {name} : {round(np.mean(ks_dict[name]), 2)} +-{round(np.std(ks_dict[name]), 2)}')
    if time_dict is not None:
        print()
        print("Times")
        for name in time_dict.keys():
            print(f'{model_name} {name} : {round(np.mean(time_dict[name]), 3)} +-{round(np.std(time_dict[name]), 3)}')
    return None


def prepare_fairness(X_pool, ovr):
    """
    compute the inference of the data 'X_pool' by the model 'ovr' and retrieve the inference and the contamination rate
    """
    X0 = X_pool[X_pool[:,-1] == -1]
    X1 = X_pool[X_pool[:,-1] == 1]
    ps = np.array([len(X0), len(X1)])/len(X_pool)
    y_prob_dict = dict()
    y_prob_dict[0] = ovr.predict_proba(X0[:,:-1])
    y_prob_dict[1] = ovr.predict_proba(X1[:,:-1])
    return y_prob_dict, ps


def unfairness(data1, data2):
    """
    compute the unfairness of two populations
    """
    K = int(np.max((np.max(data1), np.max(data2))))+1
    nu_0 = np.zeros(K)
    nu_1 = np.zeros(K)

    pos, counts = np.unique(data1, return_counts=True)
    nu_0[pos.astype(int)] = counts/len(data1)

    pos, counts = np.unique(data2, return_counts=True)
    nu_1[pos.astype(int)] = counts/len(data2)
    
    #unfair_value = np.abs(nu_0 - nu_1).sum()
    unfair_value = np.abs(nu_0 - nu_1).max()
    return unfair_value

    

#------------------------------------------------------------#
# two Fairness methods
#------------------------------------------------------------#

# method 1 : under misclassification risk (computation of lambda)
def fair_hard_max(X_test, X_pool, ovr, sigma = 10**(-5)):
    start_time = time.time()

    # computation of lambda (soft and hard)
    y_prob_dict, ps = prepare_fairness(X_pool, ovr)

    def lam_fairness_hard(lam):
        res = 0
        for s in [0,1]:
            val = y_prob_dict[s]*ps[s] - (2*s-1)*lam
            res += np.mean(np.max(val, axis=1)) # softmax or np.max
        return(res)
    
    try:
        n_classes = ovr.n_classes_
    except:
        n_classes = len(ovr.classes_)

    lam_hard = optCE(fun = lam_fairness_hard, n = 2000, d = n_classes, eps = 0.001, max_iter = 100, print_results = False)

    # inference with and without fairness
    index_0 = np.where(X_test[:,-1] == -1)[0]
    index_1 = np.where(X_test[:,-1] == 1)[0]

    y_probs = ovr.predict_proba(X_test[:,:-1])
    y_preds = ovr.predict(X_test[:,:-1])

    y_prob_fair_hard = np.zeros(y_probs.shape)
    y_prob_fair_hard[index_0] = ps[0]*y_probs[index_0] - (-1)*lam_hard
    y_prob_fair_hard[index_1] = ps[1]*y_probs[index_1] - 1*lam_hard

    y_pred_fair_hard = np.argmax(y_prob_fair_hard, axis = 1)
    
    # track the time
    time_hard = time.time() - start_time

    return y_pred_fair_hard, y_prob_fair_hard, index_0, index_1, y_preds, y_probs, time_hard
    

def fair_soft_max(X_test, X_pool, ovr, c = 0.1, opt = "SAGD", sigma = 10**(-5), epsilon_fair = 0.1, print_lambda=False):
    """
    for the optimization technique we have "CE" (cross-entropy) or "SAGD" (smoothed accelerated GD)
    """
    start_time = time.time()

    # computation of lambda (soft and hard)
    y_prob_dict, ps = prepare_fairness(X_pool, ovr)
    try:    
        n_classes = ovr.n_classes_
    except:
        n_classes = len(ovr.classes_)

    def lam_fairness_soft(lam, c = c):
        res = 0
        for s in [0,1]:
            val = y_prob_dict[s]*ps[s] - (2*s-1)*lam
            res += np.mean(np.sum(softmax(val/c, axis=1)*val, axis=1)) # Smooth arg max
        return(res)

    def bivar_fairness_soft(lam, n_classes = n_classes, c = c):
        res = 0
        lamb = lam[:n_classes]
        beta = lam[n_classes:]
        for s in [0,1]:
            val = y_prob_dict[s]*ps[s] - (2*s-1)*(lamb-beta)
            res += np.mean(np.sum(softmax(val/c, axis=1)*val, axis=1)) # Smooth arg max
        res += epsilon_fair * np.sum(lamb+beta)
        return(res)

    def nablaG(lam, c = c):
        res = 0
        for s in [0,1]:
            val = y_prob_dict[s]*ps[s] - (2*s-1)*lam
            softmax_val = softmax(val/c, axis=1)
            res -= (2*s-1) * np.mean( softmax_val, axis = 0) # Smooth arg max
        return(res)

    #def nablaGlam(lam, beta, epsilon = epsilon_fair, c = c):
    #    res = 0
    #    for s in [0,1]:
    #        val = y_prob_dict[s]*ps[s] - (2*s-1)*(lam-beta)
    #        softmax_val = softmax(val/c, axis=1)
    #        res -= (2*s-1) * np.mean(np.sum(softmax_val*val, axis = 0))  # Smooth arg max
    #    res += epsilon#*np.sum(lam)
    #    return(res)
    #def nablaGbeta(lam, beta, epsilon = epsilon_fair, c = c):
    #    res = 0
    #    for s in [0,1]:
    #        val = y_prob_dict[s]*ps[s] - (2*s-1)*(lam-beta)
    #        softmax_val = softmax(val/c, axis=1)
    #        res += (2*s-1) * np.mean( softmax_val, axis = 0) # Smooth arg max
    #    res += epsilon#*np.sum(beta)
    #    return(res)
    
    if opt == "CE":
        lam_soft = optCE(fun = lam_fairness_soft, n = 2000, d = n_classes, eps = 0.001, max_iter = 100, print_results = False)
        beta_soft = np.zeros(len(lam_soft))
    elif opt == "SAGD":
        lam_soft = optSAGD(nablaG, n_classes, c = c, T = 1000)
        beta_soft = np.zeros(lam_soft.shape)
    #elif opt == "SAGD_bivar":
    #    lam_soft, beta_soft = optSAGD_bivar(nablaGlam, nablaGbeta, n_classes, epsilon = epsilon_fair, c = c, T = 1000)
    #    #print(lam_soft)
    #    #print(beta_soft)
    elif opt == "optim":
        lam_soft = optSCIPY(fun = lam_fairness_soft, n_classes = n_classes)
        beta_soft = np.zeros(len(lam_soft))
    elif opt == "optim_bivar":
        lam_soft, beta_soft = optSCIPY_bivar(fun = bivar_fairness_soft, n_classes = n_classes)
    elif opt == "optim_bivar_bis":
        lam_soft, beta_soft = optSCIPY_bivar_bis(fun = bivar_fairness_soft, n_classes = n_classes)
    # inference with and without fairness
    index_0 = np.where(X_test[:,-1] == -1)[0]
    index_1 = np.where(X_test[:,-1] == 1)[0]

    y_probs = ovr.predict_proba(X_test[:,:-1])
    y_preds = ovr.predict(X_test[:,:-1])
    
    eps = np.random.uniform(0, sigma, (y_probs.shape))
    y_prob_fair_soft = np.zeros(y_probs.shape)
    y_prob_fair_soft[index_0] = ps[0]*(y_probs[index_0]+eps[index_0]) - (-1)*(lam_soft-beta_soft)
    y_prob_fair_soft[index_1] = ps[1]*(y_probs[index_1]+eps[index_1]) - 1*(lam_soft-beta_soft)
    y_pred_fair_soft = np.argmax(y_prob_fair_soft, axis = 1)

    if print_lambda:
        print("lamb:", lam_soft.round(5))
        print("beta:", beta_soft.round(5))

    # track the time
    time_soft = time.time() - start_time

    return y_pred_fair_soft, y_prob_fair_soft, index_0, index_1, y_preds, y_probs, time_soft

# method 2 : under L2 risk + render each score fair constraint
def fair_each_score(X_test, X_pool, ovr, sigma = 10**(-5)):
    start_time = time.time()
    
    y_probs_1, ps_1 = prepare_fairness(X_pool[:len(X_pool)//2], ovr)
    y_probs_2, ps_2 = prepare_fairness(X_pool[len(X_pool)//2:], ovr)

    y_probs = ovr.predict_proba(X_test[:,:-1])
    y_preds = ovr.predict(X_test[:,:-1])
    
    index_0 = np.where(X_test[:,-1] == -1)[0]
    index_1 = np.where(X_test[:,-1] == 1)[0]
    
    try:
        n_classes = ovr.n_classes_
    except:
        n_classes = len(ovr.classes_)
    

    y_prob_fair = np.zeros(y_probs.shape)
    for k in range(n_classes):
        eps = np.random.uniform(-sigma, sigma, len(y_probs))
        y_prob_fair[:,k][index_0] = ECDF(y_probs_2[0][:,k])(y_probs[:,k][index_0] + eps[index_0])
        y_prob_fair[:,k][index_1] = ECDF(y_probs_2[1][:,k])(y_probs[:,k][index_1] + eps[index_1])
        res = 0
        for s in [0,1]:
            res += ps_1[s] * np.quantile(y_probs_1[s][:,k], q=y_prob_fair[:,k])
        y_prob_fair[:,k] = res

    y_pred_fair = np.argmax(y_prob_fair, axis=1)

    # track the time
    time_2 = time.time() - start_time
    
    return y_pred_fair, y_prob_fair, index_0, index_1, y_preds, y_probs, time_2



#------------------------------------------------------------#
# experimentations and analysis of multi-class fairness
#------------------------------------------------------------#

def run_fairness_experimentation(clf, X, y, X_pool=None, n_times = 30, print_results = True, c = 0.01, soft_opt = "optim_bivar_bis", compute_hard = False, do_ovr = True, sigma = 10**(-5), epsilon_fair = 0, compute_baseline=False, X_train_full=None, y_train_full=None, print_lambda=False):
    """
    fairness under misclassification risk (computation of lambda)
    opt in ["CE", "SADG", "SAGD_bivar"]
    """
    ks_dict = dict()
    ks_dict["unfair"]    = []
    ks_dict["fair_soft"] = []

    acc_dict = dict()
    acc_dict["unfair"]    = []
    acc_dict["fair_soft"] = []

    time_dict = dict()
    time_dict["fair_soft"] = []
    if compute_hard:
        ks_dict["fair_hard"] = []
        acc_dict["fair_hard"] = []
        time_dict["fair_hard"] = []

    n_classes = len(np.unique(y))
    if n_classes == 2 and compute_baseline:
        ks_dict["fair_baseline"] = []
        acc_dict["fair_baseline"] = []
        time_dict["fair_baseline"] = []

    for i in range(n_times):
        if i%10==0 and print_results:
            print("ite :", i)

        # train-test-split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # define the ovr strategy and fit the model
        start_time = time.time()
        if do_ovr:
            clf_multi = OneVsRestClassifier(clf).fit(X_train[:, :-1], y_train)
        else:
            clf_multi = clf.fit(X_train[:, :-1], y_train)

        training_time = time.time() - start_time
        # inference with and without fairness
        if X_pool is not None:
            if i != 0:
                print_lambda=False
            y_pred_fair_soft, y_prob_fair_soft, index_0, index_1, y_preds, y_probs, time_soft = fair_soft_max(X_test, X_pool, clf_multi, c = c, opt = soft_opt, sigma=sigma, epsilon_fair=epsilon_fair, print_lambda=print_lambda)
        else:
            if i != 0:
                print_lambda=False
            y_pred_fair_soft, y_prob_fair_soft, index_0, index_1, y_preds, y_probs, time_soft = fair_soft_max(X_test, X_train[:, :-1], clf_multi, c = c, opt = soft_opt, sigma=sigma, epsilon_fair=epsilon_fair, print_lambda=print_lambda)
        
        if compute_hard:
            if X_pool is not None:
                y_pred_fair_hard, y_prob_fair_hard, index_0, index_1, y_preds, y_probs, time_hard = fair_hard_max(X_test, X_pool, clf_multi, sigma=sigma)
            else:
                y_pred_fair_hard, y_prob_fair_hard, index_0, index_1, y_preds, y_probs, time_hard = fair_hard_max(X_test, X_train[:, :-1], clf_multi, sigma=sigma)
            ks_dict["fair_hard"].append( unfairness(y_pred_fair_hard[index_0], y_pred_fair_hard[index_1]) )
            acc_dict["fair_hard"].append( accuracy_score(y_test, y_pred_fair_hard) )
            time_dict["fair_hard"].append(time_hard+training_time)
        else:
            y_pred_fair_hard = None, 
            y_prob_fair_hard = None

        # keep the results
        ks_dict["unfair"].append( unfairness(y_preds[index_0], y_preds[index_1]) )
        ks_dict["fair_soft"].append( unfairness(y_pred_fair_soft[index_0], y_pred_fair_soft[index_1]) )
        acc_dict["unfair"].append( accuracy_score(y_test, y_preds) )
        acc_dict["fair_soft"].append( accuracy_score(y_test, y_pred_fair_soft) )
        time_dict["fair_soft"].append(time_soft+training_time)
        
        if n_classes==2 and compute_baseline:
            # train-test-split
            X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.3)
            start_time = time.time()
            mitigator = ExponentiatedGradient(clf, DemographicParity())
            mitigator.fit(X_train[:,:-1], y_train, sensitive_features=X_train[:,-1]) # instead of X_train and y_train
            y_pred_mitigated = mitigator.predict(X_test[:,:-1])
            ks_dict["fair_baseline"].append( unfairness(y_pred_mitigated[index_0], y_pred_mitigated[index_1]) )
            acc_dict["fair_baseline"].append( accuracy_score(y_test, y_pred_mitigated) )
            time_dict["fair_baseline"].append(time.time() - start_time)

    if print_results:
        print()
        print_fairness_results(acc_dict, ks_dict, time_dict)
    
    return acc_dict, ks_dict, time_dict, index_0, index_1, y_preds, y_probs, y_pred_fair_hard, y_pred_fair_soft, y_prob_fair_hard, y_prob_fair_soft


def analysis_argmax_fairness(param = "n_classes",
                             param_range = np.arange(5, 25, 5),
                             n = 200,
                             n_features = 20,
                             n_classes = 6,
                             n_clusters = 3,
                             n_pool = 400,
                             p = 0.7,
                             n_times = 20,
                             model = LogisticRegression(),
                             c = 0.01,
                             print_ites = True,
                             compute_hard = True,
                             draw_pool_in_train = False,
                             do_ovr = False):
    """
    param can be equal to one of these values :
    
    - "n_classes" (analysis of robustness to dimensions)
    - "c" (analysis of softmax efficiencies)
    - "p" (analysis of data synthetic's fairness)
    
    """
    
    make_data_args = {
        "n"          : n,
        "n_features" : n_features,
        "n_classes"  : n_classes,
        "n_clusters" : n_clusters,
        "n_pool"     : n_pool,
        "p"          : p
    }
    
    if compute_hard:
        fair_methods = ["unfair", "fair_hard", "fair_soft"]
    else:
        fair_methods = ["unfair", "fair_soft"]

    kss_classes = dict()
    accs_classes = dict()
    times_classes = dict()

    for fair_name in fair_methods:
        kss_classes[fair_name] = {"mean":[], "std":[]}
        accs_classes[fair_name] = {"mean":[], "std":[]}
        if fair_name != "unfair":
            times_classes[fair_name] = {"mean":[], "std":[]}
    
    if param in make_data_args:
        for parameter in param_range:
            if print_ites:
                print(parameter)

            make_data_args[param] = parameter

            X, y, X_pool = make_unfair_poolclassif(**make_data_args)

            if draw_pool_in_train:
                if n_pool == 0 or X_pool is None:
                    X_pool = None
                else:
                    X_pool[:n_pool,]
                X = X[n_pool:,]
                y = y[n_pool:,]

            fair_exp_args = {
                "clf" : model,
                "X" : X,
                "y" : y,
                "X_pool" : X_pool,
                "n_times" : n_times,
                "print_results" : False,
                "c" : c,
                "do_ovr": do_ovr,
                "compute_hard" : compute_hard # True
            }

            accs, kss, times, ind0, ind1, yd, yb, ydfh, ydfs, ybfh, ybfs = run_fairness_experimentation(**fair_exp_args)

            for fair_name in accs:
                kss_classes[fair_name]["mean"].append(np.mean(kss[fair_name]))
                accs_classes[fair_name]["mean"].append(np.mean(accs[fair_name]))
                kss_classes[fair_name]["std"].append(np.std(kss[fair_name]))
                accs_classes[fair_name]["std"].append(np.std(accs[fair_name]))
                if fair_name != "unfair":
                    times_classes[fair_name]["mean"].append(np.mean(times[fair_name]))
                    times_classes[fair_name]["std"].append(np.std(times[fair_name]))
    else:
        X, y, X_pool = make_unfair_poolclassif(**make_data_args)

        if draw_pool_in_train:
            if n_pool == 0 or X_pool is None:
                X_pool = None
            else:
                X_pool[:n_pool,]
            X = X[n_pool:,]
            y = y[n_pool:,]

        fair_exp_args = {
            "clf" : model,
            "X" : X,
            "y" : y,
            "X_pool" : X_pool,
            "n_times" : n_times,
            "print_results" : False,
            "c" : c,
            "do_ovr": do_ovr,
            "compute_hard" : compute_hard # True
        }
        
        for parameter in param_range:
            if print_ites:
                print(parameter)
            fair_exp_args[param] = parameter
        
            accs, kss, times, ind0, ind1, yd, yb, ydfh, ydfs, ybfh, ybfs = run_fairness_experimentation(**fair_exp_args)

            for fair_name in fair_methods:
                kss_classes[fair_name]["mean"].append(np.mean(kss[fair_name]))
                accs_classes[fair_name]["mean"].append(np.mean(accs[fair_name]))
                kss_classes[fair_name]["std"].append(np.std(kss[fair_name]))
                accs_classes[fair_name]["std"].append(np.std(accs[fair_name]))
                if fair_name != "unfair":
                    times_classes[fair_name]["mean"].append(np.mean(times[fair_name]))
                    times_classes[fair_name]["std"].append(np.std(times[fair_name]))
    
    return kss_classes, accs_classes, times_classes


def run_fair_each_score_experimentation(clf, X, y, X_pool, sigma = 10**(-5), n_times = 30, print_results=True, do_ovr = True):
    """
    render each score faire (L2 risk)
    """
    ks_dict = dict()
    ks_dict["unfair"] = []
    ks_dict["fair"] = []

    acc_dict = dict()
    acc_dict["unfair"] = []
    acc_dict["fair"] = []

    time_dict = dict()
    time_dict["fair"] = []

    for i in range(n_times):
        if i%10 == 0 and print_results:
            print("ite :", i)
            
        # split train-test-pool
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # define the ovr strategy and fit the model
        if do_ovr:
            clf_multi = OneVsRestClassifier(clf).fit(X_train[:, :-1], y_train)
        else:
            clf_multi = clf.fit(X_train[:, :-1], y_train)
        
        # fairness algorithm
        y_pred_fair, y_prob_fair, index_0, index_1, y_preds, y_probs, time_2 = fair_each_score(X_test, X_pool, clf_multi, sigma = sigma)
        
        # keep the results
        ks_dict["unfair"].append( unfairness(y_preds[index_0], y_preds[index_1]) )
        ks_dict["fair"].append( unfairness(y_pred_fair[index_0], y_pred_fair[index_1]) )
        acc_dict["unfair"].append( accuracy_score(y_test, y_preds) )
        acc_dict["fair"].append( accuracy_score(y_test, y_pred_fair) )
        time_dict["fair"].append(time_2)
    
    if print_results:
        print()
        print_fairness_results(acc_dict, ks_dict, time_dict)
        
    return acc_dict, ks_dict, time_dict, index_0, index_1, y_preds, y_probs, y_pred_fair, y_prob_fair


def run_twomethods_experimentation(clf, X, y, X_pool, n_times = 30, print_results = True, c = 0.005, do_ovr = True, sigma = 10**(-5)):
    """
    fairness under misclassification risk
    """
    ks_dict = dict()
    ks_dict["unfair"] = []
    ks_dict["argmax-fair"] = []
    ks_dict["score-fair"] = []

    acc_dict = dict()
    acc_dict["unfair"] = []
    acc_dict["argmax-fair"] = []
    acc_dict["score-fair"] = []

    time_dict = dict()
    time_dict["argmax-fair"] = []
    time_dict["score-fair"] = []

    n_classes = len(np.unique(y))

    for i in range(n_times):
        if i%10==0 and print_results:
            print("ite :", i)

        # train-test-split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # define the ovr strategy and fit the model
        if do_ovr:
            clf_multi = OneVsRestClassifier(clf).fit(X_train[:, :-1], y_train) # instead of X_train
        else:
            clf_multi = clf.fit(X_train[:, :-1], y_train)

        # inference with and without fairness
        y_pred_fair_soft, y_prob_fair_soft, index_0, index_1, y_preds, y_probs, time_soft = fair_soft_max(X_test, X_pool, clf_multi, c = c, opt = "SAGD", sigma=sigma)
        
        # fairness algorithm
        y_pred_fair, y_prob_fair, index_0, index_1, y_preds, y_probs, time_2 = fair_each_score(X_test, X_pool, clf_multi, sigma = sigma)

        # keep the results
        ks_dict["unfair"].append( unfairness(y_preds[index_0], y_preds[index_1]) )
        ks_dict["argmax-fair"].append( unfairness(y_pred_fair_soft[index_0], y_pred_fair_soft[index_1]) )
        ks_dict["score-fair"].append( unfairness(y_pred_fair[index_0], y_pred_fair[index_1]) )
        
        acc_dict["unfair"].append( accuracy_score(y_test, y_preds) )
        acc_dict["argmax-fair"].append( accuracy_score(y_test, y_pred_fair_soft) )
        acc_dict["score-fair"].append( accuracy_score(y_test, y_pred_fair) )

        time_dict["argmax-fair"].append(time_soft)
        time_dict["score-fair"].append(time_2)

    if print_results:
        print()
        print_fairness_results(acc_dict, ks_dict, time_dict)
    
    return acc_dict, ks_dict, time_dict, index_0, index_1, y_preds, y_probs, y_pred_fair_soft, y_pred_fair, y_prob_fair_soft, y_prob_fair


def analysis_params_fairness(param = "n_classes",
                                param_range = np.arange(5, 25, 5),
                                n = 200,
                                n_features = 20,
                                n_classes = 6,
                                n_clusters = 3,
                                n_pool = 400,
                                p = 0.7,
                                n_times = 20,
                                model = LogisticRegression(),
                                c = 0.01,
                                print_ites = True,
                                draw_pool_in_train = False,
                                do_ovr = False,
                                compute_score_fair = False,
                                epsilon_fair = 0,
                                epsilon_fair_range = None,
                                print_lambda = False,
                                soft_opt = "optim_bivar_bis",
                                viz_synthetic_data = True):
    """
    param can be equal to one of these values :
    
    - "n_classes" (we can analyze the robustness to dimensions) in the paper it is called "K"
    - "c" (we can analyze the softmax efficiencies)
    - "p" (we can analyze the data synthetic's fairness)
    
    """
    
    make_data_args = {
        "n"          : n,
        "n_features" : n_features,
        "n_classes"  : n_classes,
        "n_clusters" : n_clusters,
        "n_pool"     : n_pool,
        "p"          : p
    }
    
    fair_methods = ["unfair"]
    if epsilon_fair_range is None:
        fair_methods.append("eps-fair")
    else:
        fair_methods += [f"eps-fair, eps={round(epsilon, 3)}" for epsilon in epsilon_fair_range]

    kss_classes = dict()
    accs_classes = dict()
    times_classes = dict()

    for fair_name in fair_methods:
        kss_classes[fair_name] = {"mean":[], "std":[]}
        accs_classes[fair_name] = {"mean":[], "std":[]}
        if fair_name != "unfair":
            times_classes[fair_name] = {"mean":[], "std":[]}
    
    if compute_score_fair:
        kss_classes["score-fair"] = {"mean":[], "std":[]}
        accs_classes["score-fair"] = {"mean":[], "std":[]}
        times_classes["score-fair"] = {"mean":[], "std":[]}
    
    if param in make_data_args:
        for parameter in param_range:
            if print_ites:
                print(parameter)

            make_data_args[param] = parameter

            X, y, X_pool = make_unfair_poolclassif(**make_data_args)
            if viz_synthetic_data:
                data_viz_tsne(X, y)

            if draw_pool_in_train:
                if n_pool == 0 or X_pool is None:
                    X_pool = None
                else:
                    X_pool[:n_pool,]
                X = X[n_pool:,]
                y = y[n_pool:,]

            if epsilon_fair_range is None:
                fair_exp_args = {
                    "clf" : model,
                    "X" : X,
                    "y" : y,
                    "X_pool" : X_pool,
                    "n_times" : n_times,
                    "print_results" : False,
                    "c" : c,
                    "do_ovr": do_ovr,
                    "compute_hard" : False, # True
                    "epsilon_fair": epsilon_fair,
                    "print_lambda": print_lambda,
                    "soft_opt" : soft_opt
                }
                accs, kss, times, ind0, ind1, yd, yb, ydfh, ydfs, ybfh, ybfs = run_fairness_experimentation(**fair_exp_args)
                kss_classes["unfair"]["mean"].append(np.mean(kss["unfair"]))
                accs_classes["unfair"]["mean"].append(np.mean(accs["unfair"]))
                kss_classes["unfair"]["std"].append(np.std(kss["unfair"]))
                accs_classes["unfair"]["std"].append(np.std(accs["unfair"]))
                kss_classes["eps-fair"]["mean"].append(np.mean(kss["fair_soft"]))
                kss_classes["eps-fair"]["std"].append(np.std(kss["fair_soft"]))
                accs_classes["eps-fair"]["mean"].append(np.mean(accs["fair_soft"]))
                accs_classes["eps-fair"]["std"].append(np.std(accs["fair_soft"]))
                times_classes["eps-fair"]["mean"].append(np.mean(times["fair_soft"]))
                times_classes["eps-fair"]["std"].append(np.std(times["fair_soft"]))
            else:
                for it_fair, epsilon_fair in enumerate(epsilon_fair_range):
                    fair_exp_args = {
                        "clf" : model,
                        "X" : X,
                        "y" : y,
                        "X_pool" : X_pool,
                        "n_times" : n_times,
                        "print_results" : False,
                        "c" : c,
                        "do_ovr": do_ovr,
                        "compute_hard" : False, # True
                        "epsilon_fair": epsilon_fair,
                        "print_lambda": print_lambda,
                        "soft_opt" : soft_opt
                    }
                    accs, kss, times, ind0, ind1, yd, yb, ydfh, ydfs, ybfh, ybfs = run_fairness_experimentation(**fair_exp_args)
                    kss_classes[fair_methods[it_fair+1]]["mean"].append(np.mean(kss["fair_soft"]))
                    kss_classes[fair_methods[it_fair+1]]["std"].append(np.std(kss["fair_soft"]))
                    accs_classes[fair_methods[it_fair+1]]["mean"].append(np.mean(accs["fair_soft"]))
                    accs_classes[fair_methods[it_fair+1]]["std"].append(np.std(accs["fair_soft"]))
                    times_classes[fair_methods[it_fair+1]]["mean"].append(np.mean(times["fair_soft"]))
                    times_classes[fair_methods[it_fair+1]]["std"].append(np.std(times["fair_soft"]))
                kss_classes["unfair"]["mean"].append(np.mean(kss["unfair"]))
                accs_classes["unfair"]["mean"].append(np.mean(accs["unfair"]))
                kss_classes["unfair"]["std"].append(np.std(kss["unfair"]))
                accs_classes["unfair"]["std"].append(np.std(accs["unfair"]))

            if compute_score_fair:
                fair_exp_args2 = {
                    "clf" : model,
                    "X" : X,
                    "y" : y,
                    "X_pool" : X_pool,
                    "n_times" : n_times,
                    "print_results" : False,
                    #"c" : c,
                    "do_ovr": do_ovr#,
                    #"compute_hard" : compute_hard # True
                }
                accs2, kss2, times2, ind02, ind12, yd2, yb2, ydf2, ybf2 = run_fair_each_score_experimentation(**fair_exp_args2)

                kss_classes["score-fair"]["mean"].append(np.mean(kss2["fair"]))
                kss_classes["score-fair"]["std"].append(np.std(kss2["fair"]))
                accs_classes["score-fair"]["mean"].append(np.mean(accs2["fair"]))
                accs_classes["score-fair"]["std"].append(np.std(accs2["fair"]))
                times_classes["score-fair"]["mean"].append(np.mean(times2["fair"]))
                times_classes["score-fair"]["std"].append(np.std(times2["fair"]))
            
    else:
        X, y, X_pool = make_unfair_poolclassif(**make_data_args)
        if viz_synthetic_data:
            data_viz_tsne(X, y)

        if draw_pool_in_train:
            if n_pool == 0 or X_pool is None:
                X_pool = None
            else:
                X_pool[:n_pool,]
            X = X[n_pool:,]
            y = y[n_pool:,]

        fair_exp_args = {
            "clf" : model,
            "X" : X,
            "y" : y,
            "X_pool" : X_pool,
            "n_times" : n_times,
            "print_results" : False,
            "c" : c,
            "do_ovr" : do_ovr,
            "epsilon_fair" : epsilon_fair,
            "print_lambda": print_lambda,
            "soft_opt" : soft_opt
        }
        if compute_score_fair:
            fair_exp_args2 = {
                    "clf" : model,
                    "X" : X,
                    "y" : y,
                    "X_pool" : X_pool,
                    "n_times" : n_times,
                    "print_results" : False,
                    #"c" : c,
                    "do_ovr": do_ovr#,
                    #"compute_hard" : compute_hard # True
                }
        
        for parameter in param_range:
            if print_ites:
                print(parameter)
            fair_exp_args[param] = parameter
        
            accs, kss, times, ind0, ind1, yd, yb, ydfh, ydfs, ybfh, ybfs = run_fairness_experimentation(**fair_exp_args)
            if compute_score_fair:
                accs2, kss2, times2, ind02, ind12, yd2, yb2, ydf2, ybf2 = run_fair_each_score_experimentation(**fair_exp_args2)

            kss_classes["unfair"]["mean"].append(np.mean(kss["unfair"]))
            accs_classes["unfair"]["mean"].append(np.mean(accs["unfair"]))
            kss_classes["unfair"]["std"].append(np.std(kss["unfair"]))
            accs_classes["unfair"]["std"].append(np.std(accs["unfair"]))

            kss_classes["eps-fair"]["mean"].append(np.mean(kss["fair_soft"]))
            kss_classes["eps-fair"]["std"].append(np.std(kss["fair_soft"]))
            accs_classes["eps-fair"]["mean"].append(np.mean(accs["fair_soft"]))
            accs_classes["eps-fair"]["std"].append(np.std(accs["fair_soft"]))
            times_classes["eps-fair"]["mean"].append(np.mean(times["fair_soft"]))
            times_classes["eps-fair"]["std"].append(np.std(times["fair_soft"]))

            if compute_score_fair:
                kss_classes["score-fair"]["mean"].append(np.mean(kss2["fair"]))
                kss_classes["score-fair"]["std"].append(np.std(kss2["fair"]))
                accs_classes["score-fair"]["mean"].append(np.mean(accs2["fair"]))
                accs_classes["score-fair"]["std"].append(np.std(accs2["fair"]))
                times_classes["score-fair"]["mean"].append(np.mean(times2["fair"]))
                times_classes["score-fair"]["std"].append(np.std(times2["fair"]))
    
    return kss_classes, accs_classes, times_classes



#------------------------------------------------------------#
# vizualisation of multi-class classification's fairness
#------------------------------------------------------------#

def viz_fairness_results(y_preds, index_0, index_1, acc_dict, ks_dict, name, add_title="", figsize=(8, 1.5)):
    """
    show the the contamination frequencies per class, along with the model accuracy and the fairness performance
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])
    x = np.arange(len(np.unique(y_preds)))
    plt.bar(np.arange(len(np.unique(y_preds[index_0]))) - 0.1, np.unique(y_preds[index_0], return_counts=True)[1]/len(y_preds[index_0]), alpha = 0.8, color = "steelblue", width=0.2)
    plt.bar(np.arange(len(np.unique(y_preds[index_1]))) + 0.1, np.unique(y_preds[index_1], return_counts=True)[1]/len(y_preds[index_1]), alpha = 0.8, color = "red", width=0.2)
    ax.set_ylabel('Density')
    ax.set_title(rf'{add_title} {name} : Acc {round(np.mean(acc_dict[name]), 3)}, $\hat U$ {round(np.mean(ks_dict[name]), 4)}')
    ax.set_xticks(x)
    ax.legend(labels=['S = -1', 'S = +1'])
    plt.show()
    return None


def add_arrow(line, position=None, direction='right', size=30, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    try:
        if position is None:
            position = ydata.mean()
        # find closest index
        start_ind = np.argmin(np.absolute(ydata - position))
        if direction == 'right':
            end_ind = start_ind + 1
        else:
            end_ind = start_ind - 1

        line.axes.annotate('',
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="->", color=color, lw = 2),
            size=size
        )
    except:
        if position is None:
            position = xdata.mean()
        # find closest index
        start_ind = np.argmin(np.absolute(xdata - position))
        if direction == 'right':
            end_ind = start_ind + 1
        else:
            end_ind = start_ind - 1

        line.axes.annotate('',
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="->", color=color, lw = 2),
            size=size
        )

def viz_fairness_analysis(param, param_range, kss_param, accs_param, times_param, pareto = False):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))

    sns.set_palette("colorblind")
    
    for fair_name in kss_param.keys():
        x = np.array(param_range)
        y = np.array(kss_param[fair_name]["mean"])
        std = np.array(kss_param[fair_name]["std"])
        #plt.plot(x, y, label=fair_name, marker='o')
        axes[0].errorbar(x, y, label=fair_name, yerr=std)
        axes[0].fill_between(x, (y-std), (y+std), alpha=.3)
    axes[0].legend()
    axes[0].set_title(f"Unfairness evaluation w.r.t {param}")
    axes[0].set_xlabel(param)
    axes[0].set_ylabel("Unfairness")

    for fair_name in accs_param.keys():
        x = np.array(param_range)
        y = np.array(accs_param[fair_name]["mean"])
        std = np.array(accs_param[fair_name]["std"])
        #plt.plot(x, y, label=fair_name, marker='|', ms = 10)
        axes[1].errorbar(x, y, label=fair_name, yerr=std)
        axes[1].fill_between(x, (y-std), (y+std), alpha=.3)
    axes[1].legend()
    axes[1].set_title(f"Model evaluation w.r.t {param}")
    axes[1].set_xlabel(param)
    axes[1].set_ylabel("Accuracy")
    
    if pareto:
        for fair_name in accs_param.keys():
            x = np.array(kss_param[fair_name]["mean"])
            y = np.array(accs_param[fair_name]["mean"])
            line = axes[2].plot(x, y, label=fair_name, lw = 2)[0]
            add_arrow(line)
            line.axes.annotate(f"{param_range[-1]}",
                xytext=(x[-1]-0.05, y[-1]-0.05),
                xy=(x[-1], y[-1]),
                size=12
            )
            color = line.get_color()
            axes[2].scatter(x[0], y[0], marker = "o", color=color)
            axes[2].scatter(x[-1], y[-1], marker = "o", color=color)
            for r in np.arange(0.2, 2, 0.2):
                axes[2].add_patch(plt.Circle((0, 1), r, linestyle = "--", color='black', fill=False, alpha=.3))
            line.axes.annotate(f"{param} = {param_range[0]}",
                    xytext=(x[0]-0.05, y[0]+0.02),
                    xy=(x[0], y[0]),
                    size=12
                )
        #line.axes.annotate(f"{param} = {param_range[0]}",
        #        xytext=(x[0]-0.05, y[0]+0.02),
        #        xy=(x[0], y[0]),
        #        size=12
        #    )
        axes[2].legend()
        axes[2].set_title(f"Phase diagram")
        axes[2].set_xlabel(r"Unfairness $\hat{\mathcal{U}}$")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_xlim([0,max(kss_param["unfair"]["mean"])+0.05])
        axes[2].set_ylim([0.4,max(accs_param["unfair"]["mean"])+0.05])
    else:
        for fair_name in times_param.keys():
            x = np.array(param_range)
            y = np.array(times_param[fair_name]["mean"])
            std = np.array(times_param[fair_name]["std"])
            #plt.plot(x, y, label=fair_name, marker='+')
            axes[2].errorbar(x, y, label=fair_name, yerr=std)
            axes[2].fill_between(x, (y-std), (y+std), alpha=.3)
        plt.legend()
        plt.title(f"time evaluatation w.r.t {param}")
        plt.xlabel(param)
        plt.ylabel("Time (sec)")
    
    plt.show()
    return None


def viz_fairness_distributions(yb, ybf, ind0, ind1):
    """
    yb : unfair inferences
    ybf : fair inferences
    ind0 : S=-1 indexes
    ind1 : S=+1 indexes
    """
    fig = plt.figure(figsize=(16, 4*yb.shape[1]))
    ax = fig.add_axes([0,0,1,1])

    N_C = yb.shape[1]
    for cl in np.arange(N_C):
        plt.subplot(N_C, 2, 2*cl+1)
        sns.kdeplot(yb[ind0, cl], shade= True, linewidth= 3)
        sns.kdeplot(yb[ind1, cl], shade= True, linewidth= 3)
        plt.legend(labels=['unfair S = -1',  'unfair S = +1'])
        plt.title(f"unfair : kde for class {cl}")
        
        plt.subplot(N_C, 2, 2*cl+2)
        sns.kdeplot(ybf[ind0, cl], shade= True, linewidth= 3)
        sns.kdeplot(ybf[ind1, cl], shade= True, linewidth= 3)
        plt.legend(labels=['S = -1', 'S = +1'])
        plt.title(f"fair : kde for class {cl}")
    return None


def viz_fairness_distributions_(yb, ybf, ind0, ind1, add_title=""):
    """
    yb : unfair inferences
    ybf : fair inferences
    ind0 : S=-1 indexes
    ind1 : S=+1 indexes
    """
    sns.set_palette("colorblind", 10)
    sns.color_palette("colorblind", 10)
    
    fig = plt.figure(figsize=(16, 4*2))
    ax = fig.add_axes([0,0,1,1])

    N_C = yb.shape[1]
    for i, cl in enumerate([1, N_C-1]):
        plt.subplot(2, 2, 2*i+1)
        sns.kdeplot(yb[ind0, cl], shade= True, linewidth= 3)
        sns.kdeplot(yb[ind1, cl], shade= True, linewidth= 3)
        plt.legend(labels=['unfair S = -1',  'unfair S = +1'])
        plt.title(f"unfair : kde for class {cl}")
        
        plt.subplot(2, 2, 2*i+2)
        sns.kdeplot(ybf[ind0, cl], shade= True, linewidth= 3)
        sns.kdeplot(ybf[ind1, cl], shade= True, linewidth= 3)
        plt.legend(labels=['S = -1', 'S = +1'])
        plt.title(f"{add_title} fair : kde for class {cl}")
    return None


def viz_fairness_distributions_uncond(yb, ybf, ind0, ind1):
    fig = plt.figure(figsize=(16, 2*yb.shape[1]))
    ax = fig.add_axes([0,0,1,1])

    N_C = yb.shape[1]

    for cl in np.arange(N_C):
        plt.subplot(N_C//2, 2, cl+1)
        sns.kdeplot(yb[:, cl], shade= True, linewidth= 3)
        sns.kdeplot(ybf[:, cl], shade= True, linewidth= 3)
        plt.legend(labels=['unfair', 'fair'])
        plt.title(f"class {cl}")
    return None


def viz_fairness_distributions_uncond_models(yb, ybf_model, ind0, ind1):
    fig = plt.figure(figsize=(16, 2*yb.shape[1]))
    ax = fig.add_axes([0,0,1,1])

    N_C = yb.shape[1]

    for cl in np.arange(N_C):
        plt.subplot(N_C//2, 2, cl+1)
        sns.kdeplot(yb[:, cl], shade= True, linewidth= 3, label = "unfair")
        for model_name in ybf_model:
            sns.kdeplot(ybf_model[model_name][:, cl], shade= True, linewidth= 3, label = f"fair ({model_name})")
        plt.legend()
        plt.title(f"class {cl}")
    return None


