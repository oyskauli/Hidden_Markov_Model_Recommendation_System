#%%
import os
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import FINNHelper
import DATAHelper
import numpy as np
import pandas as pd
import cProfile
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from joblib import Parallel, delayed

from mc_rec_sparse import MCRecommender
from mc_rec_factorized import MCRecommender_F
#from mc_rec import MCRecommender
from mc_data_func import mc_import_data, generate_data, generate_data_noise, mc_batch_data, generate_dist_mc, generate_dist_V
from mc_rec_tests import MCRecTests

c_path = #Global path to current dir
#colors
col_main = "#55aacc"
col_second = "#cc5555"
col_true = "#444499"

sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.rcParams["patch.force_edgecolor"] = False


def trasition_samp(s, pi, A, v, v_p, A_p, pi_p, rec_states, n_items, users, session_length, data_length, samples = 20):
    np.random.seed(s)
    n_states = A.shape[0]

    pi_prior = np.zeros([rec_states]) + pi_p
    A_prior = np.zeros([rec_states, rec_states]) + A_p
    v_prior = np.zeros([rec_states, n_items]) + v_p

    rec = MCRecommender(rec_states, n_items, users)
    rec.pi_prior = pi_prior
    rec.A_prior = A_prior
    rec.v_prior = v_prior

    res = np.zeros([samples, n_states, n_states], dtype = rec.dtype)

    for i in range(samples):
        print("Gen data")
        user_sessions, true_states, obs = generate_data(A, pi, v, users = users, session_size = session_length, user_data_length=data_length)
        batches = mc_batch_data(user_sessions)
        print("init")
        rec.initialize()
        
        print("Initial Lik: ", rec.log_lik(user_sessions, batches))
        rec.fit(user_sessions, batches, max_iter=3000, tol = 0.01)
        print(rec.A)

        pi_r, A_r, v_r = rec.relabel(v)
        res[i] = A_r

    return res

def train_likelihood_samp(s, pi, A, v, rec_states, user_sessions, batches, samples = 20, tol = 0.1):
    np.random.seed(s)
    n_states = A.shape[0]
    n_items = v.shape[1]
    users = len(user_sessions)

    pi_prior = np.zeros([rec_states]) + 2
    A_prior = np.zeros([rec_states, rec_states]) + 2
    v_prior = np.zeros([rec_states, n_items]) + 2

    rec = MCRecommender_F(n_states, 5, n_items, users)
    rec.pi_prior = pi_prior
    rec.A_prior = A_prior
    rec.v_prior = v_prior

    samp = np.zeros(samples)

    for i in range(samples):
        rec.initialize()
        rec.fit_parallel(user_sessions, batches, max_iter=100, tol = tol, threads=2)
        samp[i] = rec.log_lik(user_sessions, batches)

    return samp

def test_likelihood_samp(pi, A, v, rec_states, tr_user_sessions, tr_batches, te_user_sessions, te_batches, samples = 20):
    n_states = A.shape[0]
    n_items = v.shape[1]
    users = len(tr_user_sessions)

    rec = MCRecommender(n_states, n_items, users)
    rec.A = A
    rec.pi = pi
    rec.mubin_v = v
    
    te_true_lik = rec.log_lik(te_user_sessions, te_batches)
    tr_true_lik = rec.log_lik(tr_user_sessions, tr_batches)
    
    #calc pop
    item_pop = np.zeros(n_items, dtype = np.float64)+2
    for u_data in tr_user_sessions:
        for s in u_data:
            np.add.at(item_pop, s[0], s[1])

    n = np.sum(item_pop)
    item_pop = item_pop/n
    te_naive_lik = 0
    for u_data in te_user_sessions:
        for s in u_data:
            te_naive_lik += np.sum(s[1]*(np.log(item_pop[s[0]])))
    
    tr_naive_lik = 0
    for u_data in tr_user_sessions:
        for s in u_data:
            tr_naive_lik += np.sum(s[1]*(np.log(item_pop[s[0]])))

    tol = -tr_naive_lik/100000
    print("true: ", te_true_lik)
    print("naive: ", te_naive_lik)
    print(tol)
    pi_prior = np.zeros([rec_states]) + 2
    A_prior = np.zeros([rec_states, rec_states]) + 2
    v_prior = np.zeros([rec_states, n_items]) + 2
    
    rec = MCRecommender(rec_states, n_items, users)
    rec.pi_prior = pi_prior
    rec.A_prior = A_prior
    rec.v_prior = v_prior

    tr_samp = np.zeros(samples)
    te_samp = np.zeros(samples)

    for i in range(samples):
        rec.initialize()
        rec.fit(tr_user_sessions, tr_batches, max_iter=100, tol = tol)
        te_samp[i] = rec.log_lik(te_user_sessions, te_batches)
        tr_samp[i] = rec.log_lik(tr_user_sessions, tr_batches)

    return te_true_lik, te_naive_lik, te_samp, tr_samp, tr_true_lik, tr_naive_lik


def convergence(pi, A, v, rec_states, user_sessions, batches, iterations = 400, samples = 20):
    n_states = A.shape[0]
    n_items = v.shape[1]
    users = len(user_sessions)

    pi_prior = np.zeros([rec_states]) + 2
    A_prior = np.zeros([rec_states, rec_states]) + 2
    v_prior = np.zeros([rec_states, n_items]) + 2

    rec = MCRecommender(rec_states, n_items, users)
    rec.pi_prior = pi_prior
    rec.A_prior = A_prior
    rec.v_prior = v_prior

    samp = np.zeros(iterations)

    rec.initialize()
    for i in range(iterations):
        samp[i] = rec.fit(user_sessions, batches, max_iter=1, tol = 0)

    return samp

def convergence_entropy(pi, A, v, rec_states, user_sessions, batches, iterations = 400, samples = 20, init = 0):
    n_states = A.shape[0]
    n_items = v.shape[1]
    users = len(user_sessions)

    if(init == 0):
        #initialize to small similar prob, high entropy random
        pi_prior = np.zeros([rec_states]) + 2
        A_prior = np.zeros([rec_states, rec_states]) + 2
        v_prior = np.zeros([rec_states, n_items]) + 2

        rec = MCRecommender(rec_states, n_items, users)
        rec.pi_prior = pi_prior
        rec.A_prior = A_prior
        rec.v_prior = v_prior + 10
        rec.initialize()
        rec.v_prior = v_prior 
    if(init == 1):
        #itinialize to some high prob, low entropy random
        pi_prior = np.zeros([rec_states]) + 2
        A_prior = np.zeros([rec_states, rec_states]) + 2
        v_prior = np.zeros([rec_states, n_items]) + 2

        rec = MCRecommender(rec_states, n_items, users)
        rec.pi_prior = pi_prior
        rec.A_prior = A_prior
        
        v = generate_dist_V(v_prior, 0.02, 0.95)
        scale = 2/np.min(v)
        rec.v_prior = v*scale
        rec.initialize()
        rec.v_prior = np.zeros([rec_states, n_items])+ 2

    if(init == 2):
        #itinialize some correct as high prob, low entropy non-random
        pi_prior = np.zeros([rec_states]) + 2
        A_prior = np.zeros([rec_states, rec_states]) + 2
        v_prior = np.zeros([rec_states, n_items]) + 2

        rec = MCRecommender(rec_states, n_items, users)
        rec.pi_prior = pi_prior
        rec.A_prior = A_prior
        rec.v_prior = v_prior

        large_i = np.argsort(v, axis = 1)[:, -15:]
        scale = 2/np.min(v)
        print(scale)
        for i in range(rec_states):
            rec.v_prior[i, large_i[i]] = v[i, large_i[i]]*scale*2

        rec.initialize()
        rec.v_prior = np.zeros([rec_states, n_items]) + 2
    samp = np.zeros(iterations+1)
    ent = np.zeros((iterations+1, rec_states))

    v = rec.mubin_v
    v = v*np.log(v)
    ent[0] = -np.sum(v, axis = 1)

    for i in range(iterations):
        samp[i] = rec.fit(user_sessions, batches, max_iter=1, tol = 0)
        v = rec.mubin_v
        v = np.nan_to_num(v*np.log(v))
        ent[i+1] = -np.sum(v, axis = 1)
    
    samp[-1] = rec.log_lik(user_sessions, batches)

    return samp, ent
#%%

"""
######################
#Plot transition prob#
######################
n_states = 20

pi_prior = np.zeros(n_states) + 1
A_prior = np.zeros((n_states, n_states)) + 1
pi, A = generate_dist_mc(pi_prior, A_prior, 0, 0)
n_items = 400
true_v_prior = np.zeros([n_states, n_items]) + 1
v = generate_dist_V(true_v_prior, 0, 0.9)

samples=40
#res = trasition_samp(pi, A, 2, 1, 1, 9, 200, 4000, 7, [5, 10], samples=40)
res_i = Parallel(n_jobs=16)(delayed(trasition_samp)(s, pi, A, v, 2, 2, 2, 20, 400, 1000, 7, 14, samples=1) for s in range(samples))
res = np.squeeze(np.array(res_i))
f = open(c_path + "transition_hist.pickle", "wb")
pickle.dump((res, A), f)
#%%
sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.rcParams["patch.force_edgecolor"] = False
f = open(c_path+ "transition_hist.pickle", "rb")
res, A = pickle.load(f)
n_plot = 4
fig, ax = plt.subplots(n_plot, n_plot, sharex=True, sharey=True, constrained_layout=False)
for i in range(n_plot):
    for j in range(n_plot):
        ax[i, j].hist(res[:,0, i, j], range=(0, 1), bins = 20, rwidth = 1.1)
        ax[i, j].vlines(A[i, j], ymin = 0, ymax = 40, colors = "c", alpha = 0.9, linestyles="dashed")
        plt.xticks([0, 0.5, 1])
        plt.yticks([0, 40])
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Transition Probability")
plt.ylabel("Observations")
plt.savefig(c_path + 'transition_hist.png', )
"""
#%%

#######################
#Plot train likelihood#
#######################
item_p_user = 0.044#4.4
session_p_user = 14.7
session_size = 5

n_states = 50
n_users = 1600
n_items = 200

pi_prior = np.zeros(n_states) + 2
A_prior = np.zeros((n_states, n_states)) + 2
v_prior = np.zeros([n_states, n_items]) + 2
pi, A = generate_dist_mc(pi_prior, A_prior, 0, 0)
v = generate_dist_V(v_prior, 0, 0.5)

user_sessions, true_states, obs = generate_data(A, pi, v, n_users, session_size, session_p_user)
batches = mc_batch_data(user_sessions)

#calc pop
item_pop = np.zeros(n_items, dtype = np.float64)+2
for u_data in user_sessions:
    for s in u_data:
        np.add.at(item_pop, s[0], s[1])

n = np.sum(item_pop)
item_pop = item_pop/n
naive_lik = 0
for u_data in user_sessions:
    for s in u_data:
        naive_lik += np.sum(s[1]*(np.log(item_pop[s[0]])))
print(naive_lik)
tol = -naive_lik/100000 
rec = MCRecommender_F(n_states, 5, n_items, n_users)
rec.A = A
rec.pi = pi
rec.mubin_v = v

true_lik = rec.log_lik(user_sessions, batches)

#res = train_likelihood_samp(pi, A, v, 10, user_sessions, batches, samples = 40)
samples = 50
print("Start")
res = Parallel(n_jobs=2)(delayed(train_likelihood_samp)(s, pi, A, v, n_states, user_sessions, batches, 1, tol) for s in range(samples))
print("End")
res = np.squeeze(np.array(res))
f = open(c_path + "train_hist_" + str(n_users) + "u_" + str(n_states) + "s.pickle", "wb")
pickle.dump((true_lik, naive_lik, res), f)
#%%
f = open(c_path + "train_hist_1600u_50s.pickle", "rb")
true_lik, naive_lik, res =pickle.load(f)
#%%
sns.set_style("darkgrid")
n_plot = 3
plt.hist(res, bins = 20, color = col_main)
plt.vlines(res[0], ymin = 0, ymax = 17, colors = col_true)
plt.legend(["True Model", "Fit Likelihood"])
plt.savefig(c_path + 'train_hist.png')


# %%
"""
######################
#Plot test likelihood#
######################
n_states = 50
n_items = 1600

item_p_user = 0.044#4.4
session_p_user = 14.7
session_size = 5

pi_prior = np.zeros(n_states) + 2
A_prior = np.zeros((n_states, n_states)) + 2
v_prior = np.zeros([n_states, n_items]) + 2
pi, A = generate_dist_mc(pi_prior, A_prior, 0, 0)
v = generate_dist_V(v_prior, 0, 0.5)

train_user_sessions, true_states, train_obs = generate_data(A, pi, v, 800, 7, [5, 10])
train_batches = mc_batch_data(train_user_sessions)

test_user_sessions, true_states, test_obs = generate_data(A, pi, v, 800, 7, [5, 10])
test_batches = mc_batch_data(test_user_sessions)

samples = 40
res = Parallel(n_jobs=16)(delayed(test_likelihood_samp)(s, pi, A, v, 10, train_user_sessions, train_batches, test_user_sessions, test_batches, samples = 1) for s in range(samples))
#res = test_likelihood_samp(pi, A, v, 10, train_user_sessions, train_batches, test_user_sessions, test_batches, samples = 40)
res = np.squeeze(np.array(res))
f = open(c_path + "test_hist.pickle", "wb")
pickle.dump((res, train_obs), f)

#%%
f = open(c_path + "test_hist.pickle", "rb")
res, train_obs = pickle.load(f)
plt.hist(res[2], bins = 20)
plt.vlines(res[0], ymin = 0, ymax = 5, colors = "C1")
plt.vlines(res[1], ymin = 0, ymax = 5, colors = "C2")

plt.savefig(c_path + 'test2_hist.png')
"""
"""
#%%
###########################
#Plot lik toward finn data#
###########################
def _PL_L_TF(i, pi_prior, A_prior, v_prior, n_users, session_size, session_p_user):
    np.random.seed(i)
    test_users = 500
    pi, A = generate_dist_mc(pi_prior, A_prior, 0, 0)
    v = generate_dist_V(v_prior, 0.2, 0.5)
    #v = v*pop#adjust selection prob to popularity
    #v = v/np.expand_dims(np.sum(v, axis = 1), axis = -1)#normalize
    

    train_user_sessions, true_states, train_obs = generate_data(A, pi, v, n_users, session_size, session_p_user)
    train_batches = mc_batch_data(train_user_sessions)
    np.random.seed(0)
    test_user_sessions, true_states, test_obs = generate_data(A, pi, v, test_users, session_size, session_p_user)
    test_batches = mc_batch_data(test_user_sessions)
    print("testObs: ", test_obs)
    np.random.seed(i)
    res_s = test_likelihood_samp(pi, A, v, n_states, train_user_sessions, train_batches, test_user_sessions, test_batches, samples = 1)
    res_s = np.array([res_s[0], res_s[1], res_s[2], res_s[3], res_s[4], res_s[5]])
    return res_s
#Ratios with 30min split
item_p_user = 0.44
session_p_user = 14.7
session_size = 5

#load popularity dist
f = open(c_path + "pop_hist.pickle", "rb")
pop_hist = pickle.load(f)[5:]

users =  [1000, 1600, 3200, 6400, 12800]#, 20000]#data sizes to test
states = [ 50,  50,   50,   50,   50]

samples = 40

res = np.zeros([len(users), samples, 6])#6lines: true lik, naive lik, test fitted lik, train fitted lik, train true and naive lik
for i, n_users in enumerate(users):
    n_states = states[i]
    n_items = 200#int(np.floor(n_users*item_p_user))

    pi_prior = np.zeros(n_states) + 2
    A_prior = np.zeros((n_states, n_states)) + 2
    v_prior = np.zeros([n_states, n_items]) + 2

    #create pop dist
    q = np.linspace(1-1/(n_items*2), 1/(n_items*2), num=(n_items))
    pop = np.quantile(pop_hist, q)

    res_i = Parallel(n_jobs=16)(delayed(_PL_L_TF)(s, pi_prior, A_prior, v_prior, n_users, session_size, session_p_user) for s in range(samples))
    res[i] = np.array(res_i)

    f = open(c_path + "test_itemNoise_plot"+str(users[-1])+"u"+str(samples)+"s.pickle", "wb")
    pickle.dump((res, users), f)


#%%
f = open(c_path + "test_itemNoise_plot12800u40s", "rb")
res, users = pickle.load(f)
#%%
#f = open(c_path + "test_itemRed_plot1600u10scitem.pickle", "rb")
#res, users = pickle.load(f)
#%%
sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.rcParams["patch.force_edgecolor"] = False

m_res = np.median(res, axis=1)
q_res = np.quantile(res, 0.1, axis=1)
qu_res = np.quantile(res, 0.9, axis=1)
plt.plot(users, m_res[:,0]-m_res[:,0], color = col_true)
plt.plot(users, m_res[:,2]-m_res[:,0], color = col_main)
plt.plot(users, m_res[:,1]-m_res[:,0], "--", color = col_second)
plt.fill_between(users, q_res[:,2]-m_res[:,0], qu_res[:,2]-m_res[:,0], color=col_main, alpha = 0.2)
plt.fill_between(users, q_res[:,1]-m_res[:,0], qu_res[:,1]-m_res[:,0], color=col_second, alpha = 0.2)
plt.legend(["true", "fitted",  "naive"])
plt.savefig(c_path + 'test_plot3200u10scitem.png')
#%%
sns.relplot(data=res)
"""
#%%
"""
##############################
#Plot increasing states/users#
##############################
def _Pl_IN_ST(i, pi, A, v, n_states, train_user_sessions, train_batches, test_user_sessions, test_batches, samples):#inner loop can be paralellized
    np.random.seed(i)
    res_s = test_likelihood_samp(pi, A, v, n_states, train_user_sessions, train_batches, test_user_sessions, test_batches, samples = 1)
    res_s = np.array([res_s[0], res_s[1], res_s[2], res_s[3]])
    return res_s
#Ratios with 30min split
item_p_user = 4.4
session_p_user = 14.7
session_size = 5

f = open(c_path + "pop_hist.pickle", "rb")
pop_hist = pickle.load(f)

n_users = 2000
n_items = int(np.floor(n_users*item_p_user))
true_states = 700
states = [10, 50, 100, 200, 400, 700, 800]

samples = 10

pi_prior = np.zeros(true_states) + 2
A_prior = np.zeros((true_states, true_states)) + 2
v_prior = np.zeros([true_states, n_items]) + 2

pi, A = generate_dist_mc(pi_prior, A_prior, 0, 0)
v = generate_dist_V(v_prior, 0, 0.5)
#v = v*pop#adjust selection prob to popularity
#v = v/np.expand_dims(np.sum(v, axis = 1), axis = -1)#normalize


train_user_sessions, true_states, train_obs = generate_data(A, pi, v, n_users, session_size, session_p_user)
train_batches = mc_batch_data(train_user_sessions)

test_user_sessions, true_states, test_obs = generate_data(A, pi, v, n_users, session_size, session_p_user)
test_batches = mc_batch_data(test_user_sessions)

res = np.zeros([len(states), samples, 4])#3lines: true lik, naive lik, test fitted lik and train fitted lik
for i in range(len(states)):
    n_states = states[i]
    n_items = int(np.floor(n_users*item_p_user))

    #create pop dist
    #q = np.linspace(1-1/(n_items*2), 1/(n_items*2), num=(n_items))
    #pop = np.quantile(pop_hist, q)
    res_i = Parallel(n_jobs=10)(delayed(_Pl_IN_ST)(s, pi, A, v, n_states, train_user_sessions, train_batches, test_user_sessions, test_batches, 1) for s in range(samples))
    #for s in range(samples):
    #    res_s = test_likelihood_samp(pi, A, v, n_states, train_user_sessions, train_batches, test_user_sessions, test_batches, samples = 1)
    #    res_s = np.array([res_s[0], res_s[1], res_s[2], res_s[3]])
    #    res[i,s] = res_s
    res[i] = np.array(res_i)
f = open(c_path + "test_plot"+str(states[-1])+"states.pickle", "wb")
pickle.dump((res, states), f)
"""
"""
#%%
f = open(c_path + "test_plot800states.pickle", "rb")
res, users = pickle.load(f)

m_res = np.median(res, axis=1)
q_res = np.quantile(res, 0.9, axis=1)
#plt.plot(users, m_res[:,2])
plt.plot(users, m_res[:,3])
plt.plot(users, q_res[:,3], "--")
plt.legend(["test", "train"])
plt.savefig(c_path + 'test_plot3200u5s.png')
"""
"""
# %%
#######################
#Plot covergence speed#
#######################
def _Pl_CNV_SP(i, pi, A, v, n_users, rec_states, iterations, samples):#inner loop can be paralellized
    np.random.seed(i)

    user_sessions, true_states, train_obs = generate_data(A, pi, v, n_users, session_size, session_p_user)
    batches = mc_batch_data(user_sessions)

    res_s = convergence(pi, A, v, rec_states, user_sessions, batches, iterations, samples)
    return res_s
#Ratios with 30min split
item_p_user = 4.4
session_p_user = 14.4
session_size = 5

f = open(c_path + "pop_hist.pickle", "rb")
pop_hist = pickle.load(f)

users = [100, 200, 400, 800]
states = 50
#states = [10, 50, 100, 200, 400, 700, 800]

samples = 10
iterations = 300

pi_prior = np.zeros(states) + 2

A_prior = np.zeros((states, states)) + 2
pi, A = generate_dist_mc(pi_prior, A_prior, 0, 0)
#v = v*pop#adjust selection prob to popularity
#v = v/np.expand_dims(np.sum(v, axis = 1), axis = -1)#normalize

res = np.zeros([len(users), samples, iterations])#3lines: true lik, naive lik, test fitted lik and train fitted lik
for u in range(len(users)):
    n_states = states
    n_users = users[u]
    n_items = int(np.floor(n_users*item_p_user))

    v_prior = np.zeros([states, n_items]) + 2
    v = generate_dist_V(v_prior, 0, 0.5)


    #create pop dist
    q = np.linspace(1-1/(n_items*2), 1/(n_items*2), num=(n_items))
    pop = np.quantile(pop_hist, q)
    #_Pl_CNV_SP(1, pi, A, v, n_users, states, iterations, 1)
    res_i = Parallel(n_jobs=10)(delayed(_Pl_CNV_SP)(s, pi, A, v, n_users, states, iterations, 1) for s in range(samples))

    res[u] = np.array(res_i)
f = open(c_path + "datasize_plot"+str(states)+"s.pickle", "wb")
pickle.dump((res, users), f)

#%%
f = open(c_path + "datasize_plot50s.pickle", "rb")
res, series_lengths = pickle.load(f)

# %%
f = open(c_path + "serieslength_plot_100u50s.pickle", "rb")
res, series_lengths = pickle.load(f)
#%%
m_res = np.mean(res, axis=1)
plt.plot((m_res[0] -m_res[0, 0])/(m_res[0, -1]-m_res[0, 0]))
plt.plot((m_res[1] -m_res[1, 0])/(m_res[1, -1]-m_res[1, 0]))
plt.plot((m_res[2] -m_res[2, 0])/(m_res[2, -1]-m_res[2, 0]))
plt.plot((m_res[3] -m_res[3, 0])/(m_res[3, -1]-m_res[3, 0]))

#plt.plot(m_res[1]-m_res[1, 0])
#plt.plot(m_res[2]-m_res[2, 0])
#plt.plot(m_res[3]-m_res[3, 0])

plt.legend(["1", "2", "3", "4"])
plt.savefig(c_path + 'test_plot3200u5s.png')
# %%
"""
"""
##################
#Plot convergence#
##################
def _Pl_NS1(pi, A, v, n_states, train_user_sessions, train_batches, test_user_sessions, test_batches):#inner loop can be paralellized
    np.random.seed(i)
    res_s = test_likelihood_samp(pi, A, v, n_states, train_user_sessions, train_batches, test_user_sessions, test_batches, samples = 1)
    res_s = np.array([res_s[0], res_s[1], res_s[2], res_s[3]])
    return res_s

#Ratios with 30min split
item_p_user = 4.4
session_p_user = 14.4
session_size = 5

f = open(c_path + "pop_hist.pickle", "rb")
pop_hist = pickle.load(f)

users = [100, 200, 400, 800]
states = 50
#states = [10, 50, 100, 200, 400, 700, 800]

samples = 10
iterations = 300

pi_prior = np.zeros(states) + 2

A_prior = np.zeros((states, states)) + 2
pi, A = generate_dist_mc(pi_prior, A_prior, 0, 0)
#v = v*pop#adjust selection prob to popularity
#v = v/np.expand_dims(np.sum(v, axis = 1), axis = -1)#normalize

res = np.zeros([len(users), samples, iterations])#3lines: true lik, naive lik, test fitted lik and train fitted lik
for u in range(len(users)):
    n_states = states
    n_users = users[u]
    n_items = int(np.floor(n_users*item_p_user))

    v_prior = np.zeros([states, n_items]) + 2
    v = generate_dist_V(v_prior, 0, 0.5)


    #create pop dist
    q = np.linspace(1-1/(n_items*2), 1/(n_items*2), num=(n_items))
    pop = np.quantile(pop_hist, q)
    #_Pl_CNV_SP(1, pi, A, v, n_users, states, iterations, 1)
    res_i = Parallel(n_jobs=10)(delayed(_Pl_CNV_SP)(s, pi, A, v, n_users, states, iterations, 1) for s in range(samples))

    res[u] = np.array(res_i)
f = open(c_path + "datasize_plot"+str(states)+"s.pickle", "wb")
pickle.dump((res, users), f)
"""

#%%
##########################
#Plot convergence entropy#
##########################
def _Pl_CE1_(i, pi, A, v, n_users, rec_states, iterations, samples, dat = None, init = 0):#inner loop can be paralellized
    np.random.seed(i)

    if(dat is None):
        user_sessions, true_states, train_obs = generate_data(A, pi, v, n_users, session_size, session_p_user)
    else:
        user_sessions, true_states, train_obs = dat
    batches = mc_batch_data(user_sessions)

    res_s = convergence_entropy(pi, A, v, rec_states, user_sessions, batches, iterations, samples, init = init)
    return res_s

#Ratios with 30min split
item_p_user = 4.4
session_p_user = 14.4
session_size = 5

f = open(c_path + "pop_hist.pickle", "rb")
pop_hist = pickle.load(f)

users = [100, 200, 400, 800]
states = 50
#states = [10, 50, 100, 200, 400, 700, 800]

samples = 15
iterations = 50

pi_prior = np.zeros(states) + 2

A_prior = np.zeros((states, states)) + 2
pi, A = generate_dist_mc(pi_prior, A_prior, 0, 0)
#v = v*pop#adjust selection prob to popularity
#v = v/np.expand_dims(np.sum(v, axis = 1), axis = -1)#normalize


n_states = states
n_users = 300
n_items = int(np.floor(n_users*item_p_user))

v_prior = np.zeros([states, n_items]) + 2
v = generate_dist_V(v_prior, 0.05, 0.9)


#create pop dist
q = np.linspace(1-1/(n_items*2), 1/(n_items*2), num=(n_items))
pop = np.quantile(pop_hist, q)
#res = _Pl_CE1_(1, pi, A, v, n_users, states, iterations, 1)
dat = generate_data(A, pi, v, n_users, session_size, session_p_user)
res_0 = Parallel(n_jobs=samples)(delayed(_Pl_CE1_)(s, pi, A, v, n_users, states, iterations, 1, dat = dat, init = 0) for s in range(samples))
res_1 = Parallel(n_jobs=samples)(delayed(_Pl_CE1_)(s, pi, A, v, n_users, states, iterations, 1, dat = dat, init = 1) for s in range(samples))
res_2 = Parallel(n_jobs=samples)(delayed(_Pl_CE1_)(s, pi, A, v, n_users, states, iterations, 1, dat = dat, init = 2) for s in range(samples))
#%%
res = res_2
likelihoods = np.array([r[0] for r in res])
entrops = np.concatenate([r[1] for r in res], axis = 1).T
#%%
sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.rcParams["patch.force_edgecolor"] = False


m_l = np.median(likelihoods, axis=0)
q_l = np.quantile(likelihoods, 0.1, axis=0)
qu_l = np.quantile(likelihoods, 0.9, axis=0)

itr = list(range(iterations+1))
plt.plot(itr, m_l, color = col_main)
plt.fill_between(itr, q_l, qu_l, color=col_main, alpha = 0.2)
#%%
m_e = np.median(entrops, axis=0)
q_e = np.quantile(entrops, 0.1, axis=0)
qu_e = np.quantile(entrops, 0.9, axis=0)
plt.plot(itr, m_e, color = col_second)
plt.fill_between(itr, q_e, qu_e, color=col_second, alpha = 0.2)
plt.ylim([np.min(m_e[1:]) - .04, np.max(m_e+ .01)])

#%%
"""
ent = res[1]
#%%
s = 6
plt.plot(ent[:, s])
plt.ylim([ent[-1, s] - .1, np.max(ent[:, s]+ .01)])

#%%
m_ent = np.mean(ent, axis = 1)
plt.plot(m_ent)
plt.ylim([m_ent[-1] - .1, np.max(m_ent + .01)])
#%%
res[u] = np.array(res_i)

f = open(c_path + "datasize_plot"+str(states)+"s.pickle", "wb")
pickle.dump((res, users), f)
"""

"""
######################
#Plot selection noise#
######################
def _PL_L_NS1(i, pi_prior, A_prior, v_prior, p_r, n_users, session_size, session_p_user):
    np.random.seed(i)
    test_users = 500
    pi, A = generate_dist_mc(pi_prior, A_prior, 0, 0)
    v = generate_dist_V(v_prior, p_r, 0.5)
    #v = v*pop#adjust selection prob to popularity
    #v = v/np.expand_dims(np.sum(v, axis = 1), axis = -1)#normalize
    

    train_user_sessions, true_states, train_obs = generate_data(A, pi, v, n_users, session_size, session_p_user)
    train_batches = mc_batch_data(train_user_sessions)
    np.random.seed(0)
    test_user_sessions, true_states, test_obs = generate_data(A, pi, v, test_users, session_size, session_p_user)
    test_batches = mc_batch_data(test_user_sessions)
    print("testObs: ", test_obs)
    np.random.seed(i)
    res_s = test_likelihood_samp(pi, A, v, n_states, train_user_sessions, train_batches, test_user_sessions, test_batches, samples = 1)
    res_s = np.array([res_s[0], res_s[1], res_s[2], res_s[3], res_s[4], res_s[5]])
    return res_s

#Ratios with 30min split
item_p_user = 4.4
session_p_user = 14.4
session_size = 5

f = open(c_path + "pop_hist.pickle", "rb")
pop_hist = pickle.load(f)

p_r = [0, 0.1, 0.2, 0.4, 0.8]
n_states = 50
n_items = 200
n_users = 1600 
samples = 16

pi_prior = np.zeros(n_states) + 2

A_prior = np.zeros((n_states, n_states)) + 2
pi, A = generate_dist_mc(pi_prior, A_prior, 0, 0)
#v = v*pop#adjust selection prob to popularity
#v = v/np.expand_dims(np.sum(v, axis = 1), axis = -1)#normalize

res = np.zeros([len(p_r), samples, 6])#3lines: true lik, naive lik, test fitted lik and train fitted lik
for i in range(len(p_r)):

    pi_prior = np.zeros(n_states) + 2
    A_prior = np.zeros((n_states, n_states)) + 2
    v_prior = np.zeros([n_states, n_items]) + 2

    #create pop dist
    q = np.linspace(1-1/(n_items*2), 1/(n_items*2), num=(n_items))
    pop = np.quantile(pop_hist, q)

    res_i = Parallel(n_jobs=16)(delayed(_PL_L_NS1)(s, pi_prior, A_prior, v_prior, p_r[i], n_users, session_size, session_p_user) for s in range(samples))
    res[i] = np.array(res_i)

    f = open(c_path + "Noise1_plot"+str(n_states)+"s.pickle", "wb")
    pickle.dump((res, p_r), f)
"""
#%%
f = open(c_path + "Noise1_plot50s.pickle", "rb")
res, p_r = pickle.load(f)
#%%
sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.rcParams["patch.force_edgecolor"] = False

m_res = np.median(res, axis=1)
q_res = np.quantile(res, 0.1, axis=1)
qu_res = np.quantile(res, 0.9, axis=1)
plt.plot(p_r, m_res[:,0]-m_res[:,0], color = col_true)
plt.plot(p_r, m_res[:,2]-m_res[:,0], color = col_main)
plt.plot(p_r, m_res[:,1]-m_res[:,0], "--", color = col_second)
plt.fill_between(p_r, q_res[:,2]-m_res[:,0], qu_res[:,2]-m_res[:,0], color=col_main, alpha = 0.2)
plt.fill_between(p_r, q_res[:,1]-m_res[:,0], qu_res[:,1]-m_res[:,0], color=col_second, alpha = 0.2)
plt.legend(["true", "fitted",  "naive"])
plt.savefig(c_path + 'test_plot3200u10scitem.png')
# %%
"""
