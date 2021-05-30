#%%
import torch
import numpy as np

import FINNHelper
import DATAHelper

from mc_rec_sparse import MCRecommender
from mc_rec_factorized import MCRecommender_F, factorizationModel, predictionModel
from mc_data_func import mc_import_data, mc_batch_data, mc_vbatch_data, mc_import_items, sync_user_item_data, categorical_var

#############
#Data Import#
#############
sc, sqlContext = #Get pyspark cluster
lake = #Get pyspark dataset

#clicks
car_adtypes = ["20", "22", "200", "24", "26"]
user_sessions, item_map = mc_import_data(lake, item_th=100, split_th=0, session_th=10, max_length=200, return_tuple=True, adTypes=car_adtypes)

#item data
published_after = "2018-6-01 00:00:00-06"
item_dat = mc_import_items(sqlContext, published_after, car_adtypes)

#sync data
user_sessions, item_dat = sync_user_item_data(user_sessions, item_map, item_dat, threads = 1)

#Batch data
v_batches, lens = mc_vbatch_data(user_sessions, lim = 1500, min_l = 100)
#%%
#############################
#Create item data categories#
#############################
obs_year, cat_year = categorical_var(item_dat["production_year"], 0, numerical=True, min_max=(1900, 2030))
obs_make, cat_make = categorical_var(item_dat["make"], 0)
obs_fuel, cat_fuel = categorical_var(item_dat["fuel"], 0)
obs_body, cat_body = categorical_var(item_dat["body_type"], 0)

cat_vars = [obs_year, obs_make, obs_fuel, obs_body]

#%%
##############
#Create model#
##############
hidden_states = 200
N_items = len(item_dat)
N_users = len(user_sessions)
rec = MCRecommender_F(hidden_states, 50, N_items, N_users, cat_vars=cat_vars, dtype = np.float64)

print("initializing")
rec.set_prior_single(3, 3, 3, diag=8)
rec.initialize()
rec.model.penalty = 0.001

###########
#Fit model#
###########
tol = 0
n_iter = 90

v_batches = np.array(v_batches)
np.random.shuffle(v_batches)

rec.fit_parallel(0, v_batches, tol = tol, max_iter=n_iter, threads=1, low_gpu_mem=False)

model = predictionModel(rec.pi, rec.A, rec.mubin_v, item_dat)
#%%
model.script()
# %%
