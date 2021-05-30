#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
import seaborn as sns
import pickle
import copy
import FINNPlot


import sys
from mc_rec_sparse import MCRecommender, q_update, q_predict, log_lik_cuda, log_mubin_v, batch_v_v
from mc_rec_factorized import MCRecommender_F, factorizationModel, predictionModel
from mc_data_func import mc_import_data, mc_batch_data, mc_vbatch_data, mc_import_items, sync_user_item_data, categorical_var


c_path = #Global path to current dir
g_path = #Global path to current dir
o_path = #Global path to current dir
#colors
col_main = "#55aacc"
col_second = "#cc5555"
col_true = "#444499"
#%%

###############################
#Training likelihood histogram#
###############################

#Shows model getting stuck in local maxima due to use of EM
#1600 users, 200 items, 50 states, 50% Zero, No noise

f = open(c_path + "train_hist_1600u_50s.pickle", "rb")
true_lik, naive_lik, res =pickle.load(f)
sns.set_style("darkgrid")
n_plot = 3
plt.hist(res, bins = 30, color = col_main)
plt.vlines(res[0], ymin = 0, ymax = 10, colors = col_true)
plt.xlabel("Training Set Likelihod")
plt.ylabel("Observations")
plt.legend(["True Model", "Fit Likelihood"])
plt.savefig(c_path + 'train_hist.png')

# %%

##################################
#Transition probability histogram#
##################################

#Shows the transition probability estimates of the relabeled fitted models
#1000 users, 400 items, 10 states, 90% zero, No noise

sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.rcParams["patch.force_edgecolor"] = False
f = open(c_path+ "transition_hist10_400.pickle", "rb")
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
plt.savefig(c_path + 'transition_hist.png')

#%%
##############################
#Test Lik increaing data size#
##############################

#Show increasing tes lik when large amount of data availible
#200items, 50states, 50% zero, No noise

f = open(c_path + "test_item200_plot3200u50state.pickle", "rb")
res, users = pickle.load(f)
sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.rcParams["patch.force_edgecolor"] = False

m_res = np.median(res, axis=1)
q_res = np.quantile(res, 0.1, axis=1)
qu_res = np.quantile(res, 0.9, axis=1)
plt.plot(users, m_res[:,0], color = col_true)
plt.plot(users, m_res[:,2], color = col_main)
plt.plot(users, m_res[:,1], "--", color = col_second)
plt.fill_between(users, q_res[:,2], qu_res[:,2], color=col_main, alpha = 0.2)
plt.fill_between(users, q_res[:,1], qu_res[:,1], color=col_second, alpha = 0.2)
plt.xlabel("Users")
plt.ylabel("Likelihood")
plt.legend(["true", "fitted",  "naive"])
plt.savefig(c_path + 'test_plot3200u10scitem.png')

#%%
###############################################
#Increasing data with constatn user/item ratio#
###############################################

#show model behaviour as data increases with ovserved user/item ration over 5 days 0.44
#50states, 50% zero, No noise

f = open(c_path + "test_item0.44_plot3200u40s.pickle", "rb")
res, users = pickle.load(f)
sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.rcParams["patch.force_edgecolor"] = False

m_res = np.median(res, axis=1)
q_res = np.quantile(res, 0.1, axis=1)
qu_res = np.quantile(res, 0.9, axis=1)
plt.plot(users, m_res[:,0], color = col_true)
plt.plot(users, m_res[:,2], color = col_main)
plt.plot(users, m_res[:,1], "--", color = col_second)
plt.fill_between(users, q_res[:,2], qu_res[:,2], color=col_main, alpha = 0.2)
plt.fill_between(users, q_res[:,1], qu_res[:,1], color=col_second, alpha = 0.2)
plt.xlabel("Users")
plt.ylabel("Likelihood")
plt.legend(["true", "fitted",  "naive"])
plt.savefig(c_path + 'test_plot3200u10scitem.png')


# %%
##########################################################
#Increasing data with constant user/item ratio NORMALIZED#
##########################################################

#show model behaviour as data increases with ovserved user/item ration over 5 days 0.44
#50states, 50% zero, No noise

f = open(c_path + "test_item0.44_plot3200u40s.pickle", "rb")
res, users = pickle.load(f)
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
plt.xlabel("Users")
plt.ylabel("Normalized Likelihood")
plt.legend(["true", "fitted",  "naive"])
plt.savefig(c_path + 'test_plot3200u10scitem.png')

# %%
#########################
#Increasing Random Noize#
#########################

#1600users, 200items, 50states, 50%zero

f = open(c_path + "Noise1_plot50s.pickle", "rb")
res, p_r = pickle.load(f)
sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.rcParams["patch.force_edgecolor"] = False

m_res = np.median(res, axis=1)
q_res = np.quantile(res, 0.1, axis=1)
qu_res = np.quantile(res, 0.9, axis=1)
plt.plot(p_r, m_res[:,0], color = col_true)
plt.plot(p_r, m_res[:,2], color = col_main)
plt.plot(p_r, m_res[:,1], "--", color = col_second)
plt.fill_between(p_r, q_res[:,2], qu_res[:,2], color=col_main, alpha = 0.2)
plt.fill_between(p_r, q_res[:,1], qu_res[:,1], color=col_second, alpha = 0.2)
plt.xlabel("Probability of random selection")
plt.ylabel("Likelihood")
plt.legend(["true", "fitted",  "naive"])
plt.savefig(c_path + 'test_plot3200u10scitem.png')

# %%
####################################
#Increasing Random Noize NORMALIZED#
####################################

#1600users, 200items, 50states, 50%zero

f = open(c_path + "Noise1_plot50s.pickle", "rb")
res, p_r = pickle.load(f)
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
plt.xlabel("Probability of random selection")
plt.ylabel("Normalized Likelihood")
plt.legend(["true", "fitted",  "naive"])
plt.savefig(c_path + 'test_plot3200u10scitem.png')

# %%
############################
#Plot Noise Increasing Data#
############################

#50states, 50%zero, p=0.2 random noise

f = open(c_path + "test_itemNoise_plot12800u40s.pickle", "rb")
res, users = pickle.load(f)
sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.rcParams["patch.force_edgecolor"] = False

m_res = np.median(res, axis=1)
q_res = np.quantile(res, 0.1, axis=1)
qu_res = np.quantile(res, 0.9, axis=1)
plt.plot(users, m_res[:,0], color = col_true)
plt.plot(users, m_res[:,2], color = col_main)
plt.plot(users, m_res[:,1], "--", color = col_second)
plt.fill_between(users, q_res[:,2], qu_res[:,2], color=col_main, alpha = 0.2)
plt.fill_between(users, q_res[:,1], qu_res[:,1], color=col_second, alpha = 0.2)
plt.xlabel("Users")
plt.ylabel("Likelihood")
plt.legend(["true", "fitted",  "naive"])
plt.savefig(c_path + 'test_plot3200u10scitem.png')


#%%
#####################
#Training likelihood#
#####################
f = open(g_path + "ParamOptim/factorized_convergence_cardata_wdata_noprior.pickle", "rb")
a = pickle.load(f)
sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
sns.set_theme()

a = a[0]
a = a[1:np.min(np.where(a == 0))]

plt.plot(a)
plt.xlabel("Iteration")
plt.ylabel("Likelihood")
plt.title("Training set Likelihood")
plt.savefig(c_path + "training_likelihood_evol.png")


#%%
###########
#Load Data#
###########
f = open(g_path + "ParamOptim/data_v_car_all_dataset.pickle", "rb")
user_sessions, validation_sessions, test_sessions, item_dat= pickle.load(f)
user_sessions = np.array(user_sessions)
test_sessions = np.array(test_sessions)
l = len(user_sessions)
user_datalengths = np.zeros(l)
i = 0
for u_data in user_sessions:
    user_datalengths[i] = len(u_data)
    i += 1

indexes = np.where(user_datalengths == 0)
user_sessions = np.delete(user_sessions, indexes)

v_batches, lens = mc_vbatch_data(test_sessions, lim = 1000, min_l = 100)

#############################
#Create item data categories#
#############################
item_dat = item_dat.reset_index(drop=True)
obs_year, cat_year = categorical_var(item_dat["production_year"], 0, numerical=True, min_max=(1900, 2030))
obs_make, cat_make = categorical_var(item_dat["make"], 0)
obs_fuel, cat_fuel = categorical_var(item_dat["fuel"], 0)
obs_body, cat_body = categorical_var(item_dat["body_type"], 0)

cat_vars = [obs_year, obs_make, obs_fuel, obs_body]


##############
#Create model#
##############
hidden_states = 300
N_items = len(item_dat)
N_users = len(user_sessions)
rec = MCRecommender_F(hidden_states, 30, N_items, N_users, cat_vars=cat_vars, dtype = np.float64)

print("initializing")
rec.set_prior_single(1, 1, 1, diag=4)
rec.initialize()
rec.model.penalty = 0.001

modelpath = g_path + "ParamOptim/Fact_dim/dim_30.pickle"
rec.load(modelpath)

# %%
############################
#Plot Transition prob match#
############################
"""
PI, A, V = rec.get_parameters(0, v_batches, threads=2)

f = open(c_path + "test_set_params.pickle", "wb")
pickle.dump([PI, A, V], f)
"""
#%%
f = open(c_path + "test_set_params.pickle", "rb")
PI, A, V = pickle.load(f)

#%%
sns.set_theme()
#sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.figure(figsize=[5, 5])
plt.scatter(A.flatten(), rec.A.flatten(), color = col_main, alpha=0.3)
plt.plot([0, 1], [0, 1], color = "000000", alpha = 0.3, linestyle = "--")
plt.xlabel("Observed transition probability")
plt.ylabel("Fitted transition probability")
plt.title("Transition Probabilities, All")
plt.savefig(c_path + 'Transition_Probabilities_All.png')

#%%
tA_nan = copy.deepcopy(A)
rA_nan = copy.deepcopy(rec.A)

np.fill_diagonal(tA_nan, np.nan)
np.fill_diagonal(rA_nan, np.nan)
plt.figure(figsize=[5, 5])
plt.scatter(tA_nan.flatten(), rA_nan.flatten(), color = col_main, alpha=0.3)
plt.plot([0, 1], [0, 1], color = "000000", alpha = 0.3, linestyle = "--")
plt.xlim([0, 0.05])
plt.ylim([0, 0.05])
plt.xlabel("Observed transition probability")
plt.ylabel("Fitted transition probability")
plt.title("Transition Probabilities, Not-Diagonal")
plt.savefig(c_path + 'Transition_Probabilities_Not_Diagonal.png')

#%%
plt.figure(figsize=[5, 5])
plt.scatter(A.diagonal(), rec.A.diagonal(), color = col_main, alpha=0.3)
plt.plot([0, 1], [0, 1], color = "000000", alpha = 0.3, linestyle = "--")
plt.xlim([0.75, 1])
plt.ylim([0.75, 1])
plt.xlabel("Observed transition probability")
plt.ylabel("Fitted transition probability")
plt.title("Transition Probabilities, Diagonal")
plt.savefig(c_path + 'Transition_Probabilities_Diagonal.png')

#%%
#########################
#Plot decreasing history#
#########################

def _lik_batch_(batches, pi, A, mubin_v, length=10):
    likelihood = 0
    #B = len(batches)
    #for b in range(B):
    batch = batches#[b]
    N = batch.shape[1] - length

    if(N <= 0):
        likelihood += log_lik_cuda(batch, 0, hidden_states, A, pi, mubin_v, f_mubin = log_mubin_v, batchv=batch_v_v)[1][-1]
        print("nodelay")
    else:
        sub_b = batch[:, -length:]
        likelihood += log_lik_cuda(sub_b, 0, hidden_states, A, pi, mubin_v, f_mubin = log_mubin_v, batchv=batch_v_v)[1][-1]

    return likelihood

N = [1, 3, 5, 8, 10, 15, 20, 25, 30, 50, 100, 200]
lik = []

batch_lens = np.zeros(len(v_batches))
for i, b in enumerate(v_batches):
    batch_lens[i] = b.shape[1]

batch_i = np.where(batch_lens > 100)[0]
batch_i = batch_i[0:30]
#%%
v_batches = np.array(v_batches)
for n in N:
    with parallel_backend("threading", n_jobs = 3):
            batch_batch = v_batches[batch_i]
            print("STARTING N: ", n)
            res = [_lik_batch_(btch, rec.pi, rec.A, rec.mubin_v, length = n) for btch in batch_batch]
            print("DONE WITH LENGTH = ", n, len(res))
            l = 0
            for r in res:
                l+=r
    lik.append(r)


f = open(c_path + "test_set_hist_dep.pickle", "wb")
pickle.dump([N, lik], f)
#%%
popularity_tr = np.zeros(len(obs_year))
for i in range(len(user_sessions)):
    np.add.at(popularity_tr, user_sessions[i], 1)

popularity_te = np.zeros(len(obs_year))
for i in v_batches[batch_i]:
    np.add.at(popularity_te, i, 1)

#%%
prob = np.nan_to_num(np.log((popularity_tr+1)/np.sum(popularity_tr)))
lik_pop = np.sum(prob*popularity_te)
f = open(c_path + "test_set_hist_dep.pickle", "wb")
pickle.dump([N, lik, lik_pop], f)
# %%
sns.set_theme()
#sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
f = open(c_path + "test_set_hist_dep.pickle", "rb")
N, lik, lik_pop = pickle.load(f)
plt.plot(N, lik)
ticks = [str(n) for n in N]
plt.xlim([0, 100])
plt.title("Likelihood vs. History Length")
plt.ylabel("Test Likelihood")
plt.xlabel("user history length provided")
plt.tight_layout()
plt.savefig("hist_len.png")


# %%
###################
#Plot session prob#
###################

sessions = [0, 1, 2, 4, 8, 9, 10, 12, 16]
for sess in sessions:
    session = test_sessions[sess]

    max_length = 25
    n_top = 2
    sample = False

    length = min(max_length, len(session))
    state_prob = np.empty([length, hidden_states])
    state_prob_pr = np.empty([length, hidden_states])
    state_prob_full = np.empty([length, hidden_states])

    top_states = np.empty([length, n_top], dtype = np.long)
    item_strings = []
    rec.pred_user_dict.pop("2", None)
    for i in range(length):
        item_strings.append(str(item_dat.iloc[session[i]]["make"]) + ": " + 
                            str(item_dat.iloc[session[i]]["model"]))# + " " +
                            #("% i" % item_dat.iloc[session[i]]["production_year"]) + "\n" + 
                            #str(item_dat.iloc[session[i]]["body_type"]))
        s = rec.pred_newState("2", session[i])
        state_prob[i] = s
        state_prob_pr[i] =  np.sum(s*rec.A.T, axis=1)
        top_states[i] = np.argsort(s)[-n_top:]
        v = np.sum(rec.mubin_v*s[:, None], axis=0)
        #FINNPlot.plot_finn_ads([item_dat["id"][0], item_dat["id"][session[i]]], max_size=30)
        if(sample):
            ind = np.arange(len(v))
            i = np.random.choice(ind, size=n_top, replace = False, p=v)
        else:
            i = np.argsort(v)[-n_top:]
        items = item_dat["id"][i].to_list(), 
        #FINNPlot.plot_finn_ads(items, max_size=30)

    state_prob_full[-1] = state_prob[-1]
    for i in range(length-1):
        t = length - i - 1
        state_prob_full[t-1] = np.sum((np.nan_to_num(rec.A/state_prob_pr[t-1]))*state_prob_full[t], axis=1)
        state_prob_full[t-1] = state_prob[t-1]*state_prob_full[t-1]


    i = np.unique(top_states)
    probs = state_prob[:, i]
    probs2 = state_prob_full[:, i]
    probs3 = state_prob_pr[:, i]

    #plt.figure(figsize=(12, 5))
    #plt.plot(probs2)
    #plt.xticks(range(len(item_strings)), item_strings, rotation="vertical", verticalalignment="top", multialignment = "right", fontstyle = "oblique", fontweight = "bold")
    #plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    #%%
    sns.set_theme()
    #sns.set_style("dark")
    sns.color_palette("mako", as_cmap=True)
    """
    fig, axs = plt.subplots(2, sharex=True, figsize=(12, 8))
    axs[0].plot(probs)
    #axs[0].xticks(range(len(item_strings)), item_strings, rotation="vertical", verticalalignment="top", multialignment = "right", fontstyle = "oblique", fontweight = "bold")
    #axs[0].yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0].set_ylim([0, 1.05])
    axs[0].title.set_text("Given Past Data Only")
    axs[1].plot(probs2)
    axs[1].set_ylim([0, 1.05])
    axs[1].title.set_text("Given Past and Future Data")
    plt.xticks(range(len(item_strings)), item_strings, rotation=40, verticalalignment="top", multialignment = "right", fontstyle = "oblique", fontweight = "bold", horizontalalignment = "right", rotation_mode = "anchor")
    #plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.suptitle("State Probabilities Over User Session")
    plt.tight_layout()
    plt.savefig("user_session" + str(sess) + ".png")
    """
    
    plt.figure(figsize=(12, 4))
    plt.plot(probs2)
    plt.xticks(range(len(item_strings)), item_strings, rotation=40, verticalalignment="top", multialignment = "right", fontstyle = "oblique", fontweight = "bold", horizontalalignment = "right", rotation_mode = "anchor")
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig("user_session_show2.png")
    

#%%
###############
#Low state num#
###############
hidden_states = 150
N_items = len(item_dat)
N_users = len(user_sessions)
rec = MCRecommender_F(150, 30, N_items, N_users, cat_vars=cat_vars, dtype = np.float64)

print("initializing")
rec.set_prior_single(1, 1, 1, diag=4)
rec.initialize()
rec.model.penalty = 0.001
modelpath = g_path + "ParamOptim/State_dim/dim_150.pickle"
rec.load(modelpath)

# %%
#########################
#Selection prob pop corr#
#########################

popularity = np.zeros(len(item_dat))
for i in range(len(user_sessions)):
    np.add.at(popularity, user_sessions[i], 1)
for i in range(len(validation_sessions)):
    np.add.at(popularity, user_sessions[i], 1)
for i in range(len(test_sessions)):
    np.add.at(popularity, user_sessions[i], 1)

indx = np.argsort(popularity)
popularity = popularity[indx]
#%%
V = copy.deepcopy(rec.mubin_v)

V_s = np.empty(V.shape)

for i in range(V.shape[1]):
    V_s[:,i] = np.sort(V[:,i])

sp = np.sum(V_s, axis = 0)
V_s = V_s[: , indx]

V_s = V_s/np.sum(V_s, axis = 0)
p0 = V_s[-1, :]

#%%
p1 = pd.DataFrame(p0)
p1 = p1.rolling(1000, center = True).mean()
#%%
#pip install statsmodels
import patsy
from patsy import dmatrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
i_none = np.where(popularity > 0)
eq1 = "bs(train, knots=(3, 5, 7, 9), degree=3, include_intercept=False)"
eq2 = "cr(train, df=5)"
tr_x = dmatrix(eq2, {"train":np.log(popularity[i_none])}, return_type = "dataframe", NA_action=patsy.NAAction(NA_types=[]))
fit1 = sm.GLM(np.log(p0[i_none]), tr_x).fit()

p_x = np.linspace(2, 10, num = 101)

pred = fit1.predict(dmatrix(eq2, {"train":p_x}, return_type = "dataframe"))

#%%
sns.set_theme()
#sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.scatter(np.log(popularity), p0, alpha = 0.1, s=5)
#plt.plot(p_x, np.exp(pred), color=col_second)
plt.plot(np.log(popularity), p1)
plt.xlabel("log popularity")
plt.ylabel("state concentration")
plt.title("Item Popularity vs. State Concentration")
plt.savefig("pop_vs_st-prob.png")
#plt.li
#plt.ylim([-0.01, 0])

# %%
############
#Model perf#
############

data = pd.read_csv(o_path +"model-hist/rec-dash.csv")
#%%
data = data.reset_index()
data = data.drop(0)
dates = data["level_0"]
dates = pd.to_datetime(dates)
timestamp = (dates - pd.Timestamp("1970-01-01"))//pd.Timedelta("1s")
data["timestamp"] = timestamp

switch = (pd.Timestamp("2021-04-23 00:00:00")- pd.Timestamp("1970-01-01"))//pd.Timedelta("1s")
#%%
data_hmm = data[data["level_3"] == "car-factorized-hmm"]
data_als = data[data["level_3"] == "finn_used_car_private_als"]

#%%
dat = data_hmm
sub = dat[dat["timestamp"] > switch]

inScreen = 0
for i in sub["inScreen"]:
    inScreen += float(i)

numClick = 0
for i in sub['#sum_vars=click']:
    numClick += float(i)

print(inScreen, numClick)
print(numClick/inScreen)


# %%
#################
#Stationary dist#
#################

pi = copy.deepcopy(rec.pi)
A = copy.deepcopy(rec.A)

pi0 = pi
for i in range(10000):
    #pi0 = np.sum(pi0*A.T, axis = 1)
    pi0 = np.matmul(pi0, A)

sns.set_theme()
#sns.set_style("dark")
sns.color_palette("mako", as_cmap=True)
plt.scatter(np.log(pi), np.log(pi0))
plt.xlim([-9, -3])
plt.ylim([-9, -3])
plt.xlabel("Initial State Probability")
plt.ylabel("Stationary Distribution Probability")
plt.title("Initial vs. Stationary Distribution")
plt.savefig("init_vs_stationary.png")
# %%
total = 0
for u in test_sessions:
    total += len(u)
# %%
