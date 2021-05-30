#%%
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import FINNHelper
import DATAHelper
import numpy as np
import pandas as pd
import math
from itertools import chain
from operator import itemgetter
import pickle
import copy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend

c_path = #Global path to current dir
#%%
def mc_import_data(data_lake, entries_th = 10, split_th = 30*60*1000, session_th = 5, item_th = 1, max_length=0, return_indexes = True, return_tuple = False, adTypes = None):
    """
    Import data in the format used for markov chain recommender
    entries_th:    thershold for the ammount of enteries a user needs to be included
    split_th:      time between clicks to split into new session
    session_th:    number of sessions a user needs to be included
    return_indexes:Encode sessions as vextors or array of items 

    returns:
    item to index dictionary
    list of arrays with sessions in vector/array format, order corresponds to user index dictionary
    """
    #Load data
    print("loading data...")

    if(not adTypes is None):
        print("rem adtype")
        dat = (data_lake.filter(F.col("adType").isin(adTypes)))
    else:
        dat = data_lake
    
    dat = (
    dat.filter(F.col("subCategory") == "NoOfPageViews_Main")
    )

    dat = dat.select(["userId","itemId","sentEpoch", "adType"])


    #Remove items with few entries
    w_p = Window.partitionBy(dat.itemId)
    dat_c = dat.withColumn("counts_i", F.count("*").over(w_p))
    dat_c = dat_c.where(dat_c.counts_i > item_th) 

    #Remove users with few entries
    w_p = Window.partitionBy(dat.userId)
    dat_c = dat_c.withColumn("counts", F.count("*").over(w_p))
    dat_c = dat_c.where(dat_c.counts > entries_th)

    #add column with time between entries
    w = Window.partitionBy(dat_c.userId).orderBy(dat_c.sentEpoch)
    dat_c = dat_c.withColumn("deltaTime", dat.sentEpoch - F.lag(dat.sentEpoch, 1).over(w))

    #Column of where to split sessions
    w = Window.partitionBy(dat_c.userId).orderBy(dat_c.sentEpoch)
    dat_c = dat_c.withColumn("sessionSplit", F.when(dat_c.deltaTime > split_th, 1).otherwise(0))


    #count number of sessions for each user
    dat_c = dat_c.withColumn("sessionCount", F.sum(dat_c.sessionSplit).over(w_p))

    #Remove users with few sessions
    dat_c = dat_c.where(dat_c.sessionCount > session_th)

    #Session number
    dat_c = dat_c.withColumn("sessionIndex", F.sum(dat_c.sessionSplit).over(w))

    #user split over max sessions
    if(max_length > 0):
        dat_c = dat_c.withColumn("userSplit", F.when((dat_c.sessionIndex % max_length) == 0, 1).otherwise(0))
    dat_c = dat_c.withColumn("userIndex", F.sum(dat_c.userSplit).over(w))

    #include repeated clicks
    if(split_th > 0):
        w_pp = Window.partitionBy([dat_c.userId, dat_c.sessionIndex, dat_c.itemId])
        dat_c = dat_c.withColumn("itemNumber", F.count(dat_c.sessionSplit).over(w_pp))

    #Remove users with few entries
    w_p = Window.partitionBy([dat_c.userId, dat_c.userIndex])
    dat_c = dat_c.withColumn("counts", F.count("*").over(w_p))
    dat_c = dat_c.where(dat_c.counts > entries_th)


    if(split_th > 0):
        dat_c = dat_c.select("userId", "itemId", "sessionIndex", "itemNumber", "userIndex", "adType").distinct()
    else:
        dat_c = dat_c.select("userId", "itemId", "sessionIndex", "userIndex", "adType").distinct()

    #%%
    #dat_c.show(100)
    #%%
    """
    print("user indexes...")
    #Adding users indexes
    dat_c = dat_c.withColumn("indexId", F.concat(dat_c.userId, dat_c.userIndex))
    indexer = (dat_c.select("indexId").distinct()
        .rdd.map(itemgetter(0)).zipWithIndex()
        .toDF(["indexId", "label"]))
    dat_c = dat_c.join(indexer, ["indexId"])
    dat_c = dat_c.drop("userId", "indexId", "userIndex").withColumnRenamed("label", "userId")
    """
    #%%
    print("item indexes...")
    #Adding items indexes
    indexer = (dat_c.select("itemId").distinct()
        .rdd.map(itemgetter(0)).zipWithIndex()
        .toDF(["itemId", "label"]))
    dat_c = dat_c.join(indexer, ["itemId"])
    #%%
    #collecting item-index mapping
    i_Id = dat_c.select("itemId", "label", "adType").distinct().toPandas()
    dat_c = dat_c.drop("itemId").withColumnRenamed("label", "itemId")
    #%%
    print("get data...")
    if(split_th > 0):
        #Grouping to list
        #dont use just group by, order must be kept
        w = Window.partitionBy([dat_c.userId, dat_c.userIndex, dat_c.sessionIndex]).orderBy(dat_c.itemId)
        dat_c = dat_c.withColumn("sessionItem", F.collect_list("itemId").over(w)).withColumn("sessionNum", F.collect_list("itemNumber").over(w))
        dat_c = dat_c.groupby(["userId", "sessionIndex"]).agg(F.max("sessionItem").alias("sessionItem"), F.max("sessionNum").alias("sessionNum"))
        #%%
        #Merge item index and count lists
        dat_c = dat_c.withColumn("sessions", F.array(["sessionItem", "sessionNum"])).select("userId", "sessions")
        dat_c = dat_c.groupby("userId").agg(F.collect_list("sessions").alias("user_data"))

    else:
        w = Window.partitionBy([dat_c.userId, dat_c.userIndex]).orderBy(dat_c.sessionIndex)
        dat_c = dat_c.withColumn("sessionItem", F.collect_list("itemId").over(w))
        dat_c = dat_c.groupby(["userId"]).agg(F.max("sessionItem").alias("user_data"))


        print("collecting data...")
        user_sessions = dat_c.toPandas()["user_data"]
        print("Done!")

    return user_sessions, i_Id

def mc_import_items(sqlContext, published_after, ad_types):

    rl_pars = ["ad_type", "post_code", "price", "main_category", "sub_category", "prod_category", "property_type", "primary_room_area", "no_of_rooms", "production_year", "make", "model", "fuel", "body_type"]
    '''
    q_cat = f"""
        select id, {', '.join(rl_pars)}
        from ad_content
        where (first_published >= '{published_after}')
        and (main_category IN ({", ".join(main_cat)}))
        """
        #        AND state = 'ACTIVATED'
    '''
    q_at= f"""
        select id, {', '.join(rl_pars)}
        from ad_content
        where (published >= '{published_after}')
        and (ad_type IN ({", ".join(ad_types)}))
        """

    dat = FINNHelper.contentdb(sqlContext, q_at)

    res = dat.toPandas()
    return res


def _index_conv_(data, di):
        l = len(data)
        t = np.empty(l, dtype=np.int32)
        x = 0
        n = 0
        k = 0
        for i, r in data.iterrows():
            if(data.iloc[k, 0] in di):
                t[k] = di[data.iloc[k, 0]]
                n += 1
            else:
                t[k] = -1
                x += 1
            if(i%10000 == 0):
                print(x, data.iloc[k, 0])
                x = 0
            k += 1
        return t, n

def sync_user_item_data(user_sessions, item_map, item_dat, threads = 15):

    di = dict(zip(item_map["itemId"], item_map["label"]))
    dat_list = np.array_split(item_dat, threads)

    if(threads>1):
        p_t = Parallel(n_jobs=threads)(delayed(_index_conv_)(dat, di) for dat in dat_list)
    else:
        p_t = [_index_conv_(dat, di) for dat in dat_list]
    #%%
    t = np.concatenate([i[0] for i in p_t])
    item_dat["i_index"] = t

    #%%
    item_dat = item_dat.loc[item_dat["i_index"] != -1]#remove non-observed items
    item_dat["n_index"] = np.arange(len(item_dat))

    
    #create array converting from old indexes to row indexes in item_dat
    conv_di = dict(zip(range(len(di)), [-1 for i in di]))
    for i in range(len(item_dat)):
        conv_di[item_dat.iloc[i, -2]] = item_dat.iloc[i, -1]

    conv = np.empty(len(di), dtype = np.int32)
    for i in range(len(di)):
        conv[i] = conv_di[i]
    
    #apply conversion to data
    for i in range(len(user_sessions)): 
        tmp = conv[user_sessions[i]]
        idx = np.where(tmp != -1)
        user_sessions[i] = tmp[idx]

    #correct idexes after item delets    
    item_dat = item_dat.reset_index(drop = True)
    
    #delete empty user sessions
    user_sessions = np.array(user_sessions)
    l = len(user_sessions)
    user_datalengths = np.zeros(l)
    i = 0
    for u_data in user_sessions:
        user_datalengths[i] = len(u_data)
        i += 1

    indexes = np.where(user_datalengths == 0)
    user_sessions = np.delete(user_sessions, indexes)
    return user_sessions, item_dat

def categorical_var(observations, levels, numerical = False, min_max = None, min_obs=100, split=5):
    if(numerical):
        o = np.array(observations)
        i_min = np.where(o < min_max[0])[0]
        i_max = np.where(o > min_max[1])[0]
        i_nan = np.where(np.isnan(o) == True)[0]

        all_nan = np.concatenate((i_min, i_max, i_nan))
        o = np.delete(o, all_nan)
        #%%
        start = min_max[0]
        next_s = start + split

        splits = []
        o = np.sort(o)

        n = 0
        for i in range(len(o)):
            if(o[i] > next_s):
                if(n > min_obs):
                    splits.append(next_s)
                next_s += split
            n += 1

        category = np.digitize(observations, splits)

        #add nan category
        category[all_nan] = len(splits) 
        splits.append("nan")
        return category, splits
    else:
        observations = observations.str.upper()
        types = pd.unique(observations)

        #count occurances of each category
        index_dict = dict(zip(types, range(len(types))))
        counts = np.zeros(len(types))
        category = np.empty(len(observations), dtype = np.int32)
        for i in range(len(observations)):
            idx = index_dict[observations[i]]
            counts[idx] += 1

        #exclude categories with few observations
        #recategorized to none
        s_i = np.argsort(counts)[::-1]
        types = types[s_i]
        counts = counts[s_i]
        exclude = np.where(counts < min_obs)[0]
        none_i = np.where(types == None)[0][0]
        index = np.empty(len(types), dtype = np.int32)
        k = 0
        for i in range(len(types)):
            if(i in exclude):
                index[i] = none_i
            else:
                index[i] = k
            k += 1

        #convert observations to index array
        index_dict = dict(zip(types, index))
        for i in range(len(observations)):
            idx = index_dict[observations[i]]  
            category[i] = idx

        #%%
        types = np.delete(types, exclude)
        return category, types

def mc_import_data_v(data_lake, entries_th = 20, split_th = 30*60*1000, session_th = 5, item_th = 1, return_indexes = True, return_tuple = False):
    """
    VERY SLOW USE non DEPR
    Import data in the format used for markov chain recommender
    entries_th:    thershold for the ammount of enteries a user needs to be included
    split_th:      time between clicks to split into new session
    session_th:    number of sessions a user needs to be included
    return_indexes:Encode sessions as vextors or array of items 

    returns:
    item to index dictionary
    user to index dictionary
    list of arrays with sessions in vector/array format, order corresponds to user index dictionary
    """
    ###########Move use sparse matric datatype for user sessions?slower vs memory imprv?############

    print("loading data...")

    if(not adTypes is None):
        dat = (data_lake.filter(F.col("adType").isin(adTypes)))
    else:
        dat = data_lake
    
    dat = (
    data_lake.filter(F.col("subCategory") == "NoOfPageViews_Main")
    )

    dat = dat.select(["userId","itemId","sentEpoch", "adType"])
    

    #Remove users with few entries
    w_p = Window.partitionBy(dat.userId)
    dat_c = dat.withColumn("counts", F.count("*").over(w_p))
    dat_c = dat_c.where(dat_c.counts > entries_th)

    #add column with time between entries
    w = Window.partitionBy(dat_c.userId).orderBy(dat_c.sentEpoch)
    dat_c = dat_c.withColumn("deltaTime", dat.sentEpoch - F.lag(dat.sentEpoch, 1).over(w))

    #Column of where to split sessions
    w = Window.partitionBy(dat_c.userId).orderBy(dat_c.sentEpoch)
    dat_c = dat_c.withColumn("sessionSplit", F.when(dat_c.deltaTime > split_th, 1).otherwise(0))


    #count number of sessions for each user
    dat_c = dat_c.withColumn("sessionCount", F.sum(dat_c.sessionSplit).over(w_p))

    #Remove users with few sessions
    dat_c = dat_c.where(dat_c.sessionCount > session_th)

    #Session number
    dat_c = dat_c.withColumn("sessionIndex", F.sum(dat_c.sessionSplit).over(w))

    #user split over max sessions
    if(max_length > 0):
        dat_c = dat_c.withColumn("userSplit", F.when((dat_c.sessionIndex % max_length) == 0, 1).otherwise(0))
    dat_c = dat_c.withColumn("userIndex", F.sum(dat_c.userSplit).over(w))

    #include repeated clicks
    if(split_th > 0):
        w_pp = Window.partitionBy([dat_c.userId, dat_c.sessionIndex, dat_c.itemId])
        dat_c = dat_c.withColumn("itemNumber", F.count(dat_c.sessionSplit).over(w_pp))

    #Remove users with few entries
    w_p = Window.partitionBy([dat_c.userId, dat_c.userIndex])
    dat_c = dat_c.withColumn("counts", F.count("*").over(w_p))
    dat_c = dat_c.where(dat_c.counts > entries_th)

    dat_c = dat_c.select("userId", "itemId", "sessionIndex", "userIndex", "adType").distinct()


    #print("Counting data")
    N = 1#dat_c.count()
    if(N > 15000000):
        print("select smaller portion of data")
        exit(1)
    dat_c = dat_c.drop("deltaTime", "counts", "sentEpoch")
    print("Collecting data")
    pd_data = DATAHelper.toPandas(dat_c)
    print("Collecting done.")

    print("Creating dictionaries")
    #Create Item/user dict
    item_dict = dict.fromkeys(pd_data["itemId"])
    for i, item in enumerate(item_dict.keys()):
        item_dict[item] = i

    user_dict = dict.fromkeys(pd_data["userId"])
    for i, item in enumerate(user_dict.keys()):
        user_dict[item] = i

    pd_data["itemId"] = pd_data["itemId"].map(item_dict)

    pd_data["userId"] = pd_data["userId"].map(user_dict)

    #Move data to vector form
    N_users = len(user_dict.keys())
    N_items = len(item_dict.keys())
    user_sessions = []

    c_user = 0
    c_session = 0
    if(not return_indexes):
        print("converting to vector...")
        session = np.zeros((pd_data.iloc[0][4]+1, N_items), dtype= np.int16)
        for i, row in pd_data.iterrows():
            if(row[0] != c_user):
                user_sessions.append(session)
                session = np.zeros((row[4]+1, N_items), dtype= np.int16)
                c_session = 0
                c_user = row[0]
            if(row[3] == 1):
                c_session += 1
            session[c_session, row[1]] += 1
        user_sessions.append(session)

    elif (return_tuple):
        print("converting to tuple list...")
        #length = pd_data.shape[0]
        #progress_step = 0.1#%
        #progress = progress_step

        session_i = []
        session_n = []
        user_data = []
        print("loop start")
        for i, row in pd_data.iterrows():
            if(row[0] != c_user):
                user_sessions.append(user_data)
                c_session = 0
                c_user = row[0]
                user_data= []
            if(row[2] == 1):
                c_session += 1
                user_data.append(np.array([session_i, session_n], dtype=np.uint32))
                session_i = []
                session_n = []
            idx = row[1]
            if(idx in session_i):

                session_n[session_i.index(idx)] += 1
            else:
                session_i.append(idx)
                session_n.append(1)

            #Print progress
            #if((i/length)*100 > progress):
            #    print("Completion: ", progress, "%")
            #    progress += progress_step
            
        user_sessions.append(user_data)

    else:
        print("converting to index list...")
        session = []
        user_data = []
        for i, row in pd_data.iterrows():
            #print(row)
            if(row[0] != c_user):
                user_sessions.append(user_data)
                c_session = 0
                c_user = row[0]
                user_data= []
            if(row[2] == 1):
                c_session += 1
                user_data.append(np.array(session, dtype=np.int32))
                session = []
            session.append(row[1])
        user_sessions.append(user_data)

    print("Data import done")
    del pd_data

    return user_sessions, item_dict

def mc_batch_data(data, lim = 10000):
    """
    Batches useer data according to user data length.

    int lim, max number of users in a batch
    """
    batches = [] #list of user indexes to use in each batch

    count = [] #size of each batch

    batch_index = {} #batch index for a given size

    for u_i, u in enumerate(data):
        l = len(u)
        if(l in batch_index):
            i = batch_index[l]
            if(count[i] < lim):
                batches[i].append(u_i)
                count[i] += 1
            else:
                batch_index[l] = len(batches)
                count.append(1)
                batches.append([u_i])
        else:

            batch_index[l] = len(batches)
            count.append(1)
            batches.append([u_i])

    return batches

def mc_vbatch_data(data, lim = 200, min_l = 0, max_r = 4):
    """
    Make batches vectorized
    lim: max batch size
    min_l: minimum lentch betfor joining
    max_r: max items that can be removed to join
    """
    batches = mc_batch_data(data, lim)
    v_batches = []
    batch_lens = np.empty(len(batches))
    k = 0
    for b in batches:
        l = len(b)
        l_b = len(data[b[0]])

        batch_lens[k] = l
        k+=1

        v_b = np.zeros((l, l_b), dtype = np.int32)

        for i in range(l):
            v_b[i] = data[b[i]]
        v_batches.append(v_b)

    cont = True
    while cont:
        cont = False
        for i, b1 in enumerate(v_batches):
            best = max_r+1
            best_i = -1
            #is batch too small
            if len(b1) < min_l:
                for j, b2 in enumerate(v_batches):
                    diff = len(b1[0])-len(b2[0])
                    #are close enough in length
                    if(diff > 0 and diff< best):
                        #can be joined
                        if(len(b1) + len(b2) < lim):
                            best = diff
                            best_i = j
            
            #found candidate to join
            if(best_i > -1):
                pass

    return v_batches, batch_lens

def vectorize_batch(batch, data, N_items):
    """
    Returns batch of data in vector form
    """

    users = len(batch)
    length = len(data[batch[0]])

    x = np.zeros((users, length, N_items), dtype = np.int8)
    session = np.zeros(N_items, dtype = np.int8)
    for u_i, u in enumerate(batch):
        for t, s in enumerate(data[u]):
            session *= 0
            np.add.at(session, s, 1)
            x[u_i, t] = session

    return x

def generate_data(A, pi, v, users = 1000, session_size = 6, user_data_length=13, vectorized = False):
    K = v.shape[0]
    N = v.shape[1]
    
    observations = 0

    states = np.array(range(K))
    items = np.array(range(N))

    session_p = 1/session_size #geometric prob for correct size
    data_p = 1/(user_data_length-1) #geometric prob for correct size

    data = []
    true_states = []

    user_data = []
    user_true_states = []

    if(not vectorized):
        for u in range(users):
            current_state = np.random.choice(states, p = pi)
            length = int(np.random.geometric(data_p, 1)+1)
            for t in range(length):
                s = np.random.geometric(session_p, 1)
                observations += s
                session_i = np.random.choice(items, s, p = v[current_state])
                
                session = np.array(np.unique(session_i, return_counts= True))
                
                user_data.append(session)
                user_true_states.append(current_state)

                current_state = np.random.choice(states, p = A[current_state])
        
            data.append(user_data)
            true_states.append(user_true_states)
            user_true_states = []
            user_data = []
    
    if(vectorized):
        data_length = user_data_length*session_size
        for u in range(users):
            current_state = np.random.choice(states, p = pi)
            u_obs = 0
            while(u_obs < data_length):
                s = np.random.geometric(session_p, 1)
                s = min(s, data_length - u_obs)
                observations += s
                u_obs += s
                session_i = np.random.choice(items, s, p = v[current_state])

                user_data.append(session_i)
                user_true_states.append(current_state)

                current_state = np.random.choice(states, p = A[current_state])


            data.append(np.concatenate(user_data, axis=0))
            true_states.append(user_true_states)
            user_true_states = []
            user_data = []
    return data, true_states, observations

def generate_data_noise(A, pi, v, users = 1000, session_size = 6, user_data_length=(5, 20)):
    K = v.shape[0]
    N = v.shape[1]
    
    observations = 0

    states = np.array(range(K))
    items = np.array(range(N))

    session_p = 1/session_size #geometric prob for correct size
    lengths = np.array(range(user_data_length[1]-user_data_length[0]))+user_data_length[0]
    data = []
    true_states = []

    user_data = []
    user_true_states = []

    for u in range(users):
        current_state = np.random.choice(states, p = pi)
        n_next_state = np.random.geometric(session_p, 1) #Observations untill switching state
        length = np.random.choice(lengths)
        for t in range(length):
            s = np.random.geometric(session_p, 1)[0]
            print(s)
            observations += s
            session_i = np.zeros(s)
            for i in range(s):
                session_i[i] = np.random.choice(items, 1, p = v[current_state])
                n_next_state -= 1
                if n_next_state == 0:
                    current_state = np.random.choice(states, p = A[current_state])
                    n_next_state = np.random.geometric(session_p, 1)
            
            session = np.array(np.unique(session_i, return_counts= True))
            
            user_data.append(session)
            user_true_states.append(current_state)
    
        data.append(user_data)
        true_states.append(user_true_states)
        user_true_states = []
        user_data = []
    return data, true_states, observations


def generate_dist_V(v_prior, p_random_item, p_v_zero):
    n_states = v_prior.shape[0]
    n_items = v_prior.shape[1]
    v = np.zeros(v_prior.shape)

    items = np.array(range(n_items), np.int32)

    #n elements in row to be zero
    i_zero = math.floor(n_items*p_v_zero)

    #generate V
    for i in range(n_states):
        v[i] = np.random.dirichlet(v_prior[i])

        j = np.random.choice(items, i_zero)
        v[i][j] = 0

        v[i] = v[i]/np.sum(v[i])

        v[i] = (1-p_random_item)*v[i] + p_random_item/n_items

    return v

def generate_dist_mc(pi_prior, A_prior, p_v_zero, p_A_zero):
    pi = np.random.dirichlet(pi_prior)

    n_states = A_prior.shape[0]
    A = np.zeros(A_prior.shape)

    states = np.array(range(n_states), dtype = np.int32)

    #n elements in row to be zero
    s_zero = math.floor(n_states*p_A_zero)

    #generate V
    for i in range(n_states):
        A[i] = np.random.dirichlet(A_prior[i])

        j = np.random.choice(states, s_zero)
        A[i][j] = 0

        A[i] = A[i]/np.sum(A[i])

    return pi, A

