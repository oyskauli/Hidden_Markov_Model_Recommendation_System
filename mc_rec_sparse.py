#%%
import torch
import numpy as np
from mc_data_func import mc_import_data, mc_batch_data, vectorize_batch
import joblib
#conda install numba
#import numba as nb
#%%
def log_mubin(v, batch, n_batch, data):
    """
    Compute unnormalized mutinomial in logspace for batch

    inputs:
    float v[n_hidden_states * n_items]: Current distribution estimate
    list[session_length] batch: indexes of users in batch
    
    returns:
    float[n_batch * n_timesteps * n_hidden_states] unnormalized log likelihood of each hidden state
    """
    hidden_states = v.shape[0]
    data_length = len(data[batch[0]])
    log_l = np.zeros([n_batch, data_length, hidden_states])
    
    for i, u in enumerate(batch):
        for t in range(data_length):
            #get user session
            u_dat = data[u][t]
            #get items selected at current time
            indx = u_dat[0]
            #get number of each item
            num = u_dat[1]
            
            tmp = np.sum(np.log(v[:, indx])*num[None, :], axis = 1)
            log_l[i, t] = tmp

    return log_l

def log_mubin_v(v, batch, n_batch, data):
    """
    Compute unnormalized mutinomial in logspace for batch

    inputs:
    float v[n_hidden_states * n_items]: Current distribution estimate
    list[session_length] batch: indexes of users in batch
    
    returns:
    float[n_batch * n_timesteps * n_hidden_states] unnormalized log likelihood of each hidden state
    """

    return np.transpose(np.log(v[:,batch[:]]), (1, 2, 0))

def log_mubin_v_torch(v, batch, n_batch, data):
    """
    Compute unnormalized mutinomial in logspace for batch

    inputs:
    float v[n_hidden_states * n_items]: Current distribution estimate
    list[session_length] batch: indexes of users in batch
    
    returns:
    float[n_batch * n_timesteps * n_hidden_states] unnormalized log likelihood of each hidden state
    """

    return torch.transpose(torch.transpose(torch.log(v[:,batch[:]]), 0, 1), 1, 2)

def log_normalize(log_vals, axis = 1):
    """
    Normalize along axis 1, and move array out of log space
    
    float log_vals[n_batch * X]

    returns:
    float[n_batch * X]
    """

    log_vals -= np.expand_dims(np.max(log_vals, axis = axis), axis = axis)
    log_vals = np.exp(log_vals)
    return log_vals/np.expand_dims(np.sum(log_vals, axis = axis), axis = axis)

def log_normalize_with_C(log_vals, axis = 1):
    """
    Normalize along axis 1, and move array out of log space, 
    returning also log of normalization constant
    
    float log_vals[n_batch * X]

    returns:
    float[n_batch * X]
    """
    C1 = np.max(log_vals, axis = axis)
    log_vals -= np.expand_dims(C1, axis = axis)
    log_vals = np.exp(log_vals)
    C2 = np.sum(log_vals, axis = axis)
    return log_vals/np.expand_dims(C2, axis = axis), C1 + np.log(C2)

def normalize_q(self, q):
    pass

def q_update(log_f, t, q_t_t,  q_current, axis = 1):
    """
    Take predicted state probability estimates and update according to observed data
    Return also normalization factor to calculate likelihood
    inputs:
    float[n_batch * n_hidden_states] q_current:  predicted state probability estimates
    list[n_batch] batch: index of users in batch

    returns:
    float[n_batch * n_hidden_states]: updated probability estimates
    """
    #Store vals in log format instead?
    #log_q = np.log(q_current) + log_f
    #print(self.log_normalize(log_q).shape)
    #return self.log_normalize_with_C(log_q, axis = axis)
    #q_t_t[:, t+1], C_t = self.q_update(log_f[:, time], q_t1_t[:, t])
    

    q_t_t[:, t+1] = np.log(q_current[:, t]) + log_f[:,t]
    C = np.max(q_t_t[:, t+1], axis = axis)
    q_t_t[:, t+1] = q_t_t[:, t+1] - np.expand_dims(C, axis = axis)
    #print(self.log_normalize(log_q).shape)
    q_t_t[:, t+1] = np.exp(q_t_t[:, t+1])
    C2 = np.sum(q_t_t[:, t+1], axis = axis)
    q_t_t[:, t+1] = q_t_t[:, t+1]/np.expand_dims(C2, axis = axis)

    C = np.sum(C) + np.sum(np.log(C2))
    return C 

def q_predict(q_t_t, t, q_t1_t, A, axis = 2):
        """
        Take current state probability estimates and predict next probability according to A
        
        inputs:
        float[n_batch * n_hidden_states] q_current: current state probability estimates

        returns:
        float[n_batch * n_hidden_states] predicted state probability estimates
        """

        q_t1_t[:, t] = np.sum(q_t_t[:, t][:, None]*A.T, axis=axis)

def q_backward(q_t_t, q_t1_t, q_t1_u, t, A):
    """
    Calculates state probabilities given future data

    inputs:
    float[n_batch * n_hidden_states] q_t_t:  probability estimates at t given data up to t
    float[n_batch * n_hidden_states] q_t1_t: probability estimates at t + 1 given data up to t
    float[n_batch * n_hidden_states] q_t1_s: probability estimates at t + 1 given all data

    returns:
    float[n_batch * n_hidden_states] probability estimates at t given all data
    """
    #Double check this/ write a test
    q_t1_u[:, t-1] = np.sum((np.nan_to_num(A/q_t1_t[:, t-1][:, None]))*q_t1_u[:, t][:, None], axis=2) #if A and q_t1_t are zero at the same place => should be zero
    q_t1_u[:, t-1] = q_t_t[:, t-1]*q_t1_u[:, t-1]

def batch_pi(q_t1_u):
    """
    Unnormalized inital state probabilites estimate, intended for a batch of users.
    This is what wee need to store for all batches to estimate pi

    float[n_batch * n_hidden_states] q_1_s: State probailites at time 1

    returns:
    float[n_batch * n_hidden_states] Unnormalized initial state probabilities.
    """

    return np.sum(q_t1_u[:, 0], axis = 0)

def batch_A(Q_t_t, Q_t1_t, Q_t1_s, A, dtype = np.float64):
    """
    Unnormalized transition probabilities, for batch of users

    inputs:
    float[n_batch * n_timesteps   * n_hidden_states] Q_t_t:  probability estimates at t given data up to t
    float[n_batch * n_timesteps-1 * n_hidden_states] Q_t1_t: probability estimates at t + 1 given data up to t
    float[n_batch * n_timesteps   * n_hidden_states] Q_t1_s: probability estimates at t + 1 given all data

    returns:
    float[n_hidden_states * n_hidden_states] Unnormalized transition probability estimates for batch
    """
    n_t = Q_t_t.shape[1]
    hidden_states = A.shape[0]
    A_tmp = np.zeros((hidden_states, hidden_states), dtype = dtype)

    for i in range(n_t - 1):
        A_tmp += np.sum((np.nan_to_num(A/np.expand_dims(Q_t1_t[:, i], axis = 1)))*np.expand_dims(Q_t_t[:, i], axis = -1)*np.expand_dims(Q_t1_s[:, i+1], axis = 1) ,axis = 0)     
    return A_tmp


def batch_v(Q_t1_s, batch, data, v_shape, dtype = np.float64):
    """
    Unnormalized multinomial probabilities intended for batch of users

    float[n_batch * n_timesteps * n_hidden_states] Q_t1_s: probability estimates at t + 1 given all data
    list[n_batch] batch: indexes for users in batch

    returns:
    float[hidden_states * items] Unnormalized multinomial probabilities for batch of users
    """
    v = np.zeros(v_shape, dtype = dtype)
    s = Q_t1_s.shape[1]

    for j, u in enumerate(batch):#u index of user within all data, j index within batch
        for i in range(s):
            u_dat = data[u][i]
            #get items selected at current time
            indx = u_dat[0]
            #get number of each item
            num = u_dat[1]
            v[:, indx] += Q_t1_s[j, i][:, None]*num
    return v


def batch_v_v(Q_t1_s, batch, data, v_shape, dtype = np.float64):
    """
    Unnormalized multinomial probabilities intended for batch of users

    float[n_batch * n_timesteps * n_hidden_states] Q_t1_s: probability estimates at t + 1 given all data
    list[n_batch] batch: indexes for users in batch

    returns:
    float[hidden_states * items] Unnormalized multinomial probabilities for batch of users
    """
    v = np.zeros(v_shape, dtype = dtype)
    h_s = v_shape[0]

    for s in range(h_s):
        np.add.at(v[s], batch, Q_t1_s[:, :, s])

    return v

def log_mubin_v_torch(v, batch, n_batch, data):
    """
    Compute unnormalized mutinomial in logspace for batch

    inputs:
    float v[n_hidden_states * n_items]: Current distribution estimate
    list[session_length] batch: indexes of users in batch
    
    returns:
    float[n_batch * n_timesteps * n_hidden_states] unnormalized log likelihood of each hidden state
    """

    return torch.transpose(torch.transpose(torch.log(v[:,batch]), 0, 1), 1, 2)

def batch_v_v_torch(Q_t1_s, batch, data, v_shape, Vs, dtype = torch.float64):
    """
    Unnormalized multinomial probabilities intended for batch of users

    float[n_batch * n_timesteps * n_hidden_states] Q_t1_s: probability estimates at t + 1 given all data
    list[n_batch] batch: indexes for users in batch

    returns:
    float[hidden_states * items] Unnormalized multinomial probabilities for batch of users
    """
    #for i in range(len(b)):
    #    Vs[:, b[i]] += q[:, i]
    #    print(b[i], q[:, i])
    Vs.index_add_(1, torch.flatten(batch), torch.transpose(torch.flatten(Q_t1_s, 0, 1), 0, 1))



def fit_batch(batch, data, hidden_states, A, pi, mubin_v, f_mubin = log_mubin, batchv = batch_v):
    n_batch = len(batch)
    users = len(batch)
    #print("Batch len: ", users, "started")
    u = len(batch[0])#data length
    q_t_t = np.zeros((users, u, hidden_states))
    q_t1_t = np.zeros((users, u-1, hidden_states))
    q_t1_u = np.zeros((users, u, hidden_states))

    #q_update(log_f, t, q_t_t,  q_current, axis = 1):

    print("Batch len: ", users, "MUBIN")
    log_f = f_mubin(mubin_v, batch, n_batch, 0)

    t = 0
    tmp = pi
    tmp = np.tile(tmp, (users, 1))
    #C = q_update(log_f, time, q_t_t, tmp)#C:log likelihood

    q_t_t[:, 0] = np.log(tmp) + log_f[:,0]
    C = np.max(q_t_t[:, 0], axis = 1)
    q_t_t[:, 0] = q_t_t[:, 0] - np.expand_dims(C, axis = 1)
    q_t_t[:, 0] = np.exp(q_t_t[:, 0])
    C2 = np.sum(q_t_t[:, 0], axis = 1)
    q_t_t[:, 0] = q_t_t[:, 0]/np.expand_dims(C2, axis = 1)
    C = np.sum(C) + np.sum(np.log(C2))
    del tmp
    print("Batch len: ", users, "FWD")
    for t in range(u-1):
        q_predict(q_t_t, t, q_t1_t, A)
        C += q_update(log_f, t, q_t_t, q_t1_t)
    print("Batch len: ", users, "BWD")  
    q_t1_u[:, u-1] = q_t_t[:, u-1]
    for i in range(u-1):
        t = u - i - 1
        q_backward(q_t_t, q_t1_t, q_t1_u, t, A)

    print("Batch len: ", users, "BPAR")

    A2 = batch_A(q_t_t, q_t1_t, q_t1_u, A)
    v2 = batchv(q_t1_u, batch, data, mubin_v.shape)
    pi2 = batch_pi(q_t1_u)
    print("Batch len: ", users, "DEL")
    del q_t_t
    del q_t1_t
    del q_t1_u
    print("Batch len: ", users, "DONE")
    return A2, v2, pi2, C


def log_normalize_torch(log_vals, axis = 1):
    """
    Normalize along axis 1, and move array out of log space
    
    float log_vals[n_batch * X]

    returns:
    float[n_batch * X]
    """

    log_vals -= torch.unsqueeze(torch.max(log_vals, axis = axis).values, axis = axis)
    log_vals = torch.exp(log_vals)
    return log_vals/torch.unsqueeze(torch.sum(log_vals, axis = axis), axis = axis)

def log_normalize_with_C_torch(log_vals, axis = 1):
    """
    Normalize along axis 1, and move array out of log space, 
    returning also log of normalization constant
    
    float log_vals[n_batch * X]

    returns:
    float[n_batch * X]
    """
    C1 = torch.max(log_vals, axis = axis).values
    log_vals -= torch.expand_dims(C1, axis = axis)
    log_vals = torch.exp(log_vals)
    C2 = torch.sum(log_vals, axis = axis)
    return log_vals/torch.unsqueeze(C2, axis = axis), C1 + torch.log(C2)


def q_update_torch(log_f, t, q_t_t,  q_current, axis = 1):
    """
    Take predicted state probability estimates and update according to observed data
    Return also normalization factor to calculate likelihood
    inputs:
    float[n_batch * n_hidden_states] q_current:  predicted state probability estimates
    list[n_batch] batch: index of users in batch

    returns:
    float[n_batch * n_hidden_states]: updated probability estimates
    """
    #Store vals in log format instead?
    #log_q = np.log(q_current) + log_f
    #print(self.log_normalize(log_q).shape)
    #return self.log_normalize_with_C(log_q, axis = axis)
    #q_t_t[:, t+1], C_t = self.q_update(log_f[:, time], q_t1_t[:, t])
    

    q_t_t[:, t+1] = torch.log(q_current[:, t]) + log_f[:,t]
    C = torch.max(q_t_t[:, t+1], axis = axis).values
    q_t_t[:, t+1] = q_t_t[:, t+1] - torch.unsqueeze(C, axis = axis)
    #print(self.log_normalize(log_q).shape)
    q_t_t[:, t+1] = torch.exp(q_t_t[:, t+1])
    C2 = torch.sum(q_t_t[:, t+1], axis = axis)
    q_t_t[:, t+1] = q_t_t[:, t+1]/torch.unsqueeze(C2, axis = axis)

    C = torch.sum(C) + torch.sum(torch.log(C2))
    return C 

def q_predict_torch(q_t_t, t, q_t1_t, A, axis = 2):
        """
        Take current state probability estimates and predict next probability according to A
        
        inputs:
        float[n_batch * n_hidden_states] q_current: current state probability estimates

        returns:
        float[n_batch * n_hidden_states] predicted state probability estimates
        """

        q_t1_t[:, t] = torch.sum(q_t_t[:, t][:, None]*A.T, axis=axis)

def q_backward_torch(q_t_t, q_t1_t, q_t1_u, t, A):
    """
    Calculates state probabilities given future data

    inputs:
    float[n_batch * n_hidden_states] q_t_t:  probability estimates at t given data up to t
    float[n_batch * n_hidden_states] q_t1_t: probability estimates at t + 1 given data up to t
    float[n_batch * n_hidden_states] q_t1_s: probability estimates at t + 1 given all data

    returns:
    float[n_batch * n_hidden_states] probability estimates at t given all data
    """
    #WITH LOW PRIORS!! convert nan to 0
    tmp = (A/q_t1_t[:, t-1][:, None])*q_t1_u[:, t][:, None]
    tmp[torch.isnan(tmp)] = 0
    q_t1_u[:, t-1] = torch.sum(tmp, axis=2) #if A and q_t1_t are zero at the same place => should be zero
    q_t1_u[:, t-1] = q_t_t[:, t-1]*q_t1_u[:, t-1]

    #WITH HIGHER PRIORS(Faster but unsafe with low priors)
    #q_t1_u[:, t-1] = torch.sum(A/q_t1_t[:, t-1][:, None]*q_t1_u[:, t][:, None], axis=2) #if A and q_t1_t are zero at the same place => should be zero
    #q_t1_u[:, t-1] = q_t_t[:, t-1]*q_t1_u[:, t-1]

def batch_pi_torch(q_t1_u):
    """
    Unnormalized inital state probabilites estimate, intended for a batch of users.
    This is what wee need to store for all batches to estimate pi

    float[n_batch * n_hidden_states] q_1_s: State probailites at time 1

    returns:
    float[n_batch * n_hidden_states] Unnormalized initial state probabilities.
    """

    return torch.sum(q_t1_u[:, 0], axis = 0)

def batch_A_torch(Q_t_t, Q_t1_t, Q_t1_s, A, As=None, dtype = np.float64):
    """
    Unnormalized transition probabilities, for batch of users

    inputs:
    float[n_batch * n_timesteps   * n_hidden_states] Q_t_t:  probability estimates at t given data up to t
    float[n_batch * n_timesteps-1 * n_hidden_states] Q_t1_t: probability estimates at t + 1 given data up to t
    float[n_batch * n_timesteps   * n_hidden_states] Q_t1_s: probability estimates at t + 1 given all data

    returns:
    float[n_hidden_states * n_hidden_states] Unnormalized transition probability estimates for batch
    """
    n_t = Q_t_t.shape[1]
    #print(Q_t_t.shape)
    #print(Q_t1_t.shape)
    #print(Q_t1_s.shape)
    #print(A.shape)
    #print(As.shape)
    hidden_states = A.shape[0]
    if(As is None):
        As = torch.zeros((hidden_states, hidden_states), requires_grad=False, device=torch.device("cuda"), dtype = torch.float)

    for i in range(n_t - 1):
        #WITH LOW PRIORS!! convert nan to 0
        #set_Q_t1_s to remove last row
        #rep A along time axis
        tmp = (A/torch.unsqueeze(Q_t1_t[:, i], axis = 1))*torch.unsqueeze(Q_t_t[:, i], axis = -1)*torch.unsqueeze(Q_t1_s[:, i+1], axis = 1)#A and Q_t1 = 0 is the situation where a transition has prob = 0 and predicted prob of transition is 0 som nan(0/0) = 0 in this case
        tmp[torch.isnan(tmp)] = 0
        As += torch.sum(tmp,axis = 0)

        #WITH HIGHER PRIORS(Faster but unsafe with low priors)
        #tmp = A/torch.unsqueeze(Q_t1_t[:, i], axis = 1)
        #tmp[tmp != tmp] = 0
        #As += torch.sum(tmp*torch.unsqueeze(Q_t_t[:, i], axis = -1)*torch.unsqueeze(Q_t1_s[:, i+1], axis = 1) ,axis = 0)

    return As

def fit_batch_cuda(batch, data, hidden_states, A, pi, mubin_v, f_mubin = log_mubin, batchv = batch_v):
    n_batch = len(batch)
    users = len(batch)
    device = torch.device("cuda")
    A = torch.tensor(A, device=device, requires_grad=False)

    #print("Batch len: ", users, "started")
    u = len(batch[0])#data length
    q_t_t = torch.empty((users, u, hidden_states), requires_grad=False, device = device)
    q_t1_t = torch.empty((users, u-1, hidden_states), requires_grad=False, device = device)
    q_t1_u = torch.empty((users, u, hidden_states), requires_grad=False, device = device)

    #q_update(log_f, t, q_t_t,  q_current, axis = 1):

    #print("Batch len: ", users, "MUBIN")
    log_f = f_mubin(mubin_v, batch, n_batch, 0)
    log_f= torch.tensor(log_f, device=device)

    t = 0
    tmp = pi
    tmp = torch.tensor(np.tile(tmp, (users, 1)), device=device)
    #C = q_update(log_f, time, q_t_t, tmp)#C:log likelihood

    q_t_t[:, 0] = torch.log(tmp) + log_f[:,0]
    C = torch.max(q_t_t[:, 0], axis = 1).values
    q_t_t[:, 0] = q_t_t[:, 0] - torch.unsqueeze(C, axis = 1)
    q_t_t[:, 0] = torch.exp(q_t_t[:, 0])
    C2 = torch.sum(q_t_t[:, 0], axis = 1)
    q_t_t[:, 0] = q_t_t[:, 0]/torch.unsqueeze(C2, axis = 1)
    C = torch.sum(C) + torch.sum(torch.log(C2))
    del tmp
    #print("Batch len: ", users, "FWD")
    for t in range(u-1):
        q_predict_torch(q_t_t, t, q_t1_t, A)
        C += q_update_torch(log_f, t, q_t_t, q_t1_t)
    #print("Batch len: ", users, "BWD")  
    q_t1_u[:, u-1] = q_t_t[:, u-1]
    for i in range(u-1):
        t = u - i - 1
        q_backward_torch(q_t_t, q_t1_t, q_t1_u, t, A)

    #print("Batch len: ", users, "BPAR")
    q_t1_u_c = q_t1_u.cpu().numpy()
    A2 = batch_A_torch(q_t_t, q_t1_t, q_t1_u, A)
    v2 = batchv(q_t1_u_c, batch, data, mubin_v.shape)
    pi2 = batch_pi(q_t1_u_c)
    #print("Batch len: ", users, "DEL")

    #print("Batch len: ", users, "DONE")
    return A2.cpu().numpy(), v2, pi2, C.cpu().numpy()

def fit_batch_cuda_seq(batch, data, hidden_states, A, pi, mubin_v, PIs, As, Vs, f_mubin = log_mubin, batchv = batch_v, dtype = torch.float64):
    n_batch = len(batch)
    users = len(batch)
    device = torch.device("cuda")

    #print("Batch len: ", users, "started")
    u = len(batch[0])#data length
    q_t_t = torch.empty((users, u, hidden_states), requires_grad=False, dtype = dtype, device = device)
    q_t1_t = torch.empty((users, u-1, hidden_states), requires_grad=False, dtype = dtype, device = device)
    q_t1_u = torch.empty((users, u, hidden_states), requires_grad=False, dtype = dtype, device = device)

    #q_update(log_f, t, q_t_t,  q_current, axis = 1):

    #print("Batch len: ", users, "MUBIN")
    log_f = f_mubin(mubin_v, batch, n_batch, 0)

    t = 0
    tmp = pi
    tmp = tmp.repeat(users, 1)
    #C = q_update(log_f, time, q_t_t, tmp)#C:log likelihood

    q_t_t[:, 0] = torch.log(tmp) + log_f[:,0]
    C = torch.max(q_t_t[:, 0], axis = 1).values
    q_t_t[:, 0] = q_t_t[:, 0] - torch.unsqueeze(C, axis = 1)
    q_t_t[:, 0] = torch.exp(q_t_t[:, 0])
    C2 = torch.sum(q_t_t[:, 0], axis = 1)
    q_t_t[:, 0] = q_t_t[:, 0]/torch.unsqueeze(C2, axis = 1)
    C = torch.sum(C) + torch.sum(torch.log(C2))
    del tmp
    #print("Batch len: ", users, "FWD")
    for t in range(u-1):
        q_predict_torch(q_t_t, t, q_t1_t, A)
        C += q_update_torch(log_f, t, q_t_t, q_t1_t)
    #print("Batch len: ", users, "BWD")  
    q_t1_u[:, u-1] = q_t_t[:, u-1]
    for i in range(u-1):
        t = u - i - 1
        q_backward_torch(q_t_t, q_t1_t, q_t1_u, t, A)

    #print("Batch len: ", users, "BPAR")
    batch_A_torch(q_t_t, q_t1_t, q_t1_u, A, As)
    batch_v_v_torch(q_t1_u, batch, data, mubin_v.shape, Vs)
    PIs += batch_pi_torch(q_t1_u)
    #print("Batch len: ", users, "DEL")

    #print(q_t_t.dtype, q_t1_t.dtype, q_t1_u.dtype, A.dtype)
    del q_t_t
    del q_t1_t
    del q_t1_u 
    return C.cpu().numpy()


def log_lik_cuda(batch, data, hidden_states, A, pi, mubin_v, f_mubin = log_mubin, batchv = batch_v):
    n_batch = len(batch)
    users = len(batch)
    device = torch.device("cuda")
    A = torch.tensor(A, device=device, requires_grad=False)

    u = len(batch[0])#data length
    q_t_t = torch.empty((users, u, hidden_states), requires_grad=False, device = device)
    q_t1_t = torch.empty((users, u-1, hidden_states), requires_grad=False, device = device)

    #q_update(log_f, t, q_t_t,  q_current, axis = 1):

    log_f = f_mubin(mubin_v, batch, n_batch, 0)
    log_f= torch.tensor(log_f, device=device)

    t = 0
    tmp = pi
    tmp = torch.tensor(np.tile(tmp, (users, 1)), device=device)
    #C = q_update(log_f, time, q_t_t, tmp)#C:log likelihood

    q_t_t[:, 0] = torch.log(tmp) + log_f[:,0]
    C = torch.max(q_t_t[:, 0], axis = 1).values
    q_t_t[:, 0] = q_t_t[:, 0] - torch.unsqueeze(C, axis = 1)
    q_t_t[:, 0] = torch.exp(q_t_t[:, 0])
    C2 = torch.sum(q_t_t[:, 0], axis = 1)
    q_t_t[:, 0] = q_t_t[:, 0]/torch.unsqueeze(C2, axis = 1)
    C = torch.sum(C) + torch.sum(torch.log(C2))
    del tmp
    for t in range(u-1):
        q_predict_torch(q_t_t, t, q_t1_t, A)
        C += q_update_torch(log_f, t, q_t_t, q_t1_t)

    del q_t_t
    del q_t1_t
    print("LLik, Batch len: ", users, "DONE")
    return C.cpu().numpy()

def state_popularity_cuda_seq(batch, data, hidden_states, A, pi, mubin_v, f_mubin = log_mubin, batchv = batch_v, dtype = torch.float64):
    n_batch = len(batch)
    users = len(batch)
    device = torch.device("cuda")

    #print("Batch len: ", users, "started")
    u = len(batch[0])#data length
    q_t_t = torch.empty((users, u, hidden_states), requires_grad=False, dtype = dtype, device = device)
    q_t1_t = torch.empty((users, u-1, hidden_states), requires_grad=False, dtype = dtype, device = device)
    q_t1_u = torch.empty((users, u, hidden_states), requires_grad=False, dtype = dtype, device = device)

    #q_update(log_f, t, q_t_t,  q_current, axis = 1):

    #print("Batch len: ", users, "MUBIN")
    log_f = f_mubin(mubin_v, batch, n_batch, 0)
    log_f= torch.tensor(log_f, device=device)

    t = 0
    tmp = pi
    tmp = torch.tensor(tmp.repeat(users, 1), device=device)
    #C = q_update(log_f, time, q_t_t, tmp)#C:log likelihood

    q_t_t[:, 0] = torch.log(tmp) + log_f[:,0]
    C = torch.max(q_t_t[:, 0], axis = 1).values
    q_t_t[:, 0] = q_t_t[:, 0] - torch.unsqueeze(C, axis = 1)
    q_t_t[:, 0] = torch.exp(q_t_t[:, 0])
    C2 = torch.sum(q_t_t[:, 0], axis = 1)
    q_t_t[:, 0] = q_t_t[:, 0]/torch.unsqueeze(C2, axis = 1)
    C = torch.sum(C) + torch.sum(torch.log(C2))
    del tmp
    #print("Batch len: ", users, "FWD")
    for t in range(u-1):
        q_predict_torch(q_t_t, t, q_t1_t, A)
        C += q_update_torch(log_f, t, q_t_t, q_t1_t)
    #print("Batch len: ", users, "BWD")  
    q_t1_u[:, u-1] = q_t_t[:, u-1]
    for i in range(u-1):
        t = u - i - 1
        q_backward_torch(q_t_t, q_t1_t, q_t1_u, t, A)

    #q_t1_u[BATCH, TIME, STATE]
    #populartiry
    pop = torch.sum(torch.sum(q_t1_u, dim = 0), dim = 0)

    q_t1_u = torch.transpose(q_t1_u, 0, 2)
    q_t1_u = torch.flatten(q_t1_u, start_dim=1)

    corr = torch.zeros((hidden_states, hidden_states), requires_grad=False, dtype = dtype, device = device)
    for i in range(hidden_states):
        for j in range(i, hidden_states):
            corr[i, j] = torch.sum(q_t1_u[i]*q_t1_u[j])

    var_t = torch.sum(torch.sum(q_t1_u*q_t1_u, dim = 0), dim = 0)

    del q_t_t
    del q_t1_t
    del q_t1_u 
    return pop.cpu().numpy(), corr.cpu().numpy(), var_t.cpu().numpy()


#%%
class MCRecommender:
    #######
    #TO DO#
    #######

    #set min priors based on machine accuracy
    #if one observation of an item could be rounded to p = 0
    #can happen if very many items

    def __init__(self, hidden_states, items, users, dtype = np.float64):
        self.mubin_v = np.zeros([hidden_states, items], dtype=dtype)
        self.pi = np.zeros(hidden_states, dtype=dtype)
        self.A = np.zeros([hidden_states, hidden_states], dtype=dtype)

        self.pi_prior = np.ones(hidden_states, dtype=dtype)
        self.A_prior = np.ones([hidden_states, hidden_states], dtype=dtype)
        self.v_prior = np.ones([hidden_states, items], dtype=dtype)

        self.hidden_states = hidden_states
        self.items = items
        self.users = users

        self.data = []
        self.batches = []

        self.n_batch = 0
        self.c_batch = 0
        self.time = 0
        
        self.dtype = dtype

        self.pred_user_dict = {}


    def q_update(self, log_f,  q_current, axis = 1):
        """
        Take predicted state probability estimates and update according to observed data
        Return also normalization factor to calculate likelihood
        inputs:
        float[n_batch * n_hidden_states] q_current:  predicted state probability estimates
        list[n_batch] batch: index of users in batch

        returns:
        float[n_batch * n_hidden_states]: updated probability estimates
        """
        #Store vals in log format instead?

        log_q = np.log(q_current) + log_f
        #print(self.log_normalize(log_q).shape)
        return self.log_normalize_with_C(log_q, axis = axis)

    def q_predict(self, q_current, axis = 2):
        """
        Take current state probability estimates and predict next probability according to A
        
        inputs:
        float[n_batch * n_hidden_states] q_current: current state probability estimates

        returns:
        float[n_batch * n_hidden_states] predicted state probability estimates
        """
        return np.sum(q_current[:, None]*self.A.T, axis=axis)

    def q_backward(self, q_t_t, q_t1_t, q_t1_s):
        """
        Calculates state probabilities given future data

        inputs:
        float[n_batch * n_hidden_states] q_t_t:  probability estimates at t given data up to t
        float[n_batch * n_hidden_states] q_t1_t: probability estimates at t + 1 given data up to t
        float[n_batch * n_hidden_states] q_t1_s: probability estimates at t + 1 given all data

        returns:
        float[n_batch * n_hidden_states] probability estimates at t given all data
        """

        #Double check this/ write a test
        q = np.sum((np.nan_to_num(self.A/q_t1_t[:, None]))*q_t1_s[:, None], axis=2) #if A and q_t1_t are zero at the same place => should be zero
        q_t_s = q_t_t*q

        return q_t_s

    def log_mubin(self, v, batch):
        """
        Compute unnormalized mutinomial in logspace for batch

        inputs:
        float v[n_hidden_states * n_items]: Current distribution estimate
        list[session_length] batch: indexes of users in batch
        
        returns:
        float[n_batch * n_timesteps * n_hidden_states] unnormalized log likelihood of each hidden state
        """
        data_length = len(self.data[batch[0]])
        log_l = np.zeros([self.n_batch, data_length, self.hidden_states])
        
        for i, u in enumerate(batch):
            for t in range(data_length):
                #get user session
                u_dat = self.data[u][t]
                #get items selected at current time
                indx = u_dat[0]
                #get number of each item
                num = u_dat[1]
                
                tmp = np.sum(np.log(v[:, indx])*num[None, :], axis = 1)
                log_l[i, t] = tmp

        return log_l


    def log_normalize(self, log_vals, axis = 1):
        """
        Normalize along axis 1, and move array out of log space
        
        float log_vals[n_batch * X]

        returns:
        float[n_batch * X]
        """

        log_vals -= np.expand_dims(np.max(log_vals, axis = axis), axis = axis)
        log_vals = np.exp(log_vals)
        return log_vals/np.expand_dims(np.sum(log_vals, axis = axis), axis = axis)

    def log_normalize_with_C(self, log_vals, axis = 1):
        """
        Normalize along axis 1, and move array out of log space, 
        returning also log of normalization constant
        
        float log_vals[n_batch * X]

        returns:
        float[n_batch * X]
        """
        C1 = np.max(log_vals, axis = axis)
        log_vals -= np.expand_dims(C1, axis = axis)
        log_vals = np.exp(log_vals)
        C2 = np.sum(log_vals, axis = axis)
        return log_vals/np.expand_dims(C2, axis = axis), C1 + np.log(C2)


    def batch_pi(self, q_1_s):
        """
        Unnormalized inital state probabilites estimate, intended for a batch of users.
        This is what wee need to store for all batches to estimate pi

        float[n_batch * n_hidden_states] q_1_s: State probailites at time 1

        returns:
        float[n_batch * n_hidden_states] Unnormalized initial state probabilities.
        """

        return np.sum(q_1_s, axis = 0)

    def batch_A(self, Q_t_t, Q_t1_t, Q_t1_s):
        """
        Unnormalized transition probabilities, for batch of users

        inputs:
        float[n_batch * n_timesteps   * n_hidden_states] Q_t_t:  probability estimates at t given data up to t
        float[n_batch * n_timesteps-1 * n_hidden_states] Q_t1_t: probability estimates at t + 1 given data up to t
        float[n_batch * n_timesteps   * n_hidden_states] Q_t1_s: probability estimates at t + 1 given all data

        returns:
        float[n_hidden_states * n_hidden_states] Unnormalized transition probability estimates for batch
        """

        A_tmp = np.zeros((self.hidden_states, self.hidden_states), dtype = self.dtype)
        n_t = Q_t_t.shape[1]

        for i in range(n_t - 1):
            q_t_t = Q_t_t[:, i]
            q_t1_t = Q_t1_t[:, i]
            q_t1_s = Q_t1_s[:, i+1]

            q_t_t = np.expand_dims(q_t_t, axis = -1)
            q_t1_t = np.expand_dims(q_t1_t, axis = 1)
            q_t1_s = np.expand_dims(q_t1_s, axis = 1)

            A_tmp += np.sum((np.nan_to_num(self.A/q_t1_t))*q_t_t*q_t1_s  ,axis = 0)
            
        return A_tmp


    def batch_v(self, Q_t1_s, batch):
        """
        Unnormalized multinomial probabilities intended for batch of users

        float[n_batch * n_timesteps * n_hidden_states] Q_t1_s: probability estimates at t + 1 given all data
        list[n_batch] batch: indexes for users in batch

        returns:
        float[hidden_states * items] Unnormalized multinomial probabilities for batch of users
        """
        v = np.zeros([self.hidden_states, self.items], dtype = self.dtype)
        s = Q_t1_s.shape[1]

        for j, u in enumerate(batch):#u index of user within all data, j index within batch
            for i in range(s):
                u_dat = self.data[u][i]
                #get items selected at current time
                indx = u_dat[0]
                #get number of each item
                num = u_dat[1]
                v[:, indx] += Q_t1_s[j, i][:, None]*num

        return v

    def row_init(self, X, prior):
        d = len(X.shape)
        if(d==2):
            rows = X.shape[0]
            for r in range(rows):
                X[r] = np.random.dirichlet(prior[r])
            X = X/np.expand_dims(np.sum(X, axis = 1), axis = -1)
        elif(d==1):
            X = np.random.dirichlet(prior)
            X = X/np.sum(X)
        else:
            exit(1)
        return X

    def row_norm(self, X):
        d = len(X.shape)
        if(d==2):
            X = X/np.expand_dims(np.sum(X, axis = 1), axis = -1)
        elif(d==1):
            X = X/np.sum(X)
        else:
            exit(1)
        return X

    def initialize(self):
        """
        sample initial values according to prior

        float add_p, added to prior, larger => larger inital values 
        """

        self.A = self.row_init(self.A, self.A_prior) 

        self.mubin_v = self.row_init(self.mubin_v, self.v_prior)

        self.pi = self.row_init(self.pi, self.pi_prior)

    def fit(self, data, batches, max_iter = 200, tol = 1):
        self.batches = batches

        self.data = data

        converged = False
        good_updates = 0
        slow_updates = 0
        log_l_prev = 1
        
        As = np.zeros([self.hidden_states, self.hidden_states], dtype=self.dtype)
        Vs = np.zeros([self.hidden_states, self.items], dtype=self.dtype)
        PIs = np.zeros([self.hidden_states], dtype=self.dtype)
        for i in range(max_iter):
            log_l = 0
            #print(self.v)
            for j, b in enumerate(batches):
                self.c_batch = j
                A2, v2, pi2, l = self.fit_batch(b)
                log_l += l
                As += A2
                Vs += v2
                PIs += pi2
                #print("current lik: ", log_l)
                
            new_A = self.row_norm(As + self.A_prior - 1)
            #print("Diff:", np.max(np.abs(self.A-new_A)))
            self.A = new_A
            self.mubin_v = self.row_norm(Vs + self.v_prior -1)
            self.pi = self.row_norm(PIs + self.pi_prior -1)
            As = As*0
            Vs = Vs*0
            PIs = PIs*0
            print("Total lik: ", log_l)
            if(log_l_prev < 0):#not first iter
                #(log_l - log_l_prev)
                #if((log_l - log_l_prev) < tol):
                if(np.abs(log_l_prev - log_l) < tol):
                    slow_updates += 1
                    #print(slow_updates, good_updates)
                    if(slow_updates > 5 and good_updates > 5):
                        converged = True
                        break
                else:
                    slow_updates = 0
                if(log_l_prev > log_l and np.abs(log_l_prev - log_l) > tol/10):
                    good_updates = 0
                    print("OOPS")
                else:
                    good_updates += 1

            log_l_prev = log_l

        #if(not converged):
            #print("Did not converge!!")

        return log_l

    
    def fit_batch(self, batch):
        self.n_batch = len(batch)
        users = len(batch)
        u = len(self.data[batch[0]])#data length
        q_t_t = np.zeros((users, u, self.hidden_states))
        q_t1_t = np.zeros((users, u-1, self.hidden_states))
        q_t1_u = np.zeros((users, u, self.hidden_states))

        log_f = self.log_mubin(self.mubin_v, batch)

        time = 0
        tmp = self.pi
        tmp = np.tile(tmp, (users, 1))
        q_t_t[:, 0], C = self.q_update(log_f[:, time], tmp)

        for t in range(u-1):
            time = t + 1
            q_t1_t[:, t] = self.q_predict(q_t_t[:, t])
            q_t_t[:, t+1], C_t = self.q_update(log_f[:, time], q_t1_t[:, t])
            C += C_t
            
        q_t1_u[:, u-1] = q_t_t[:, u-1]
        for i in range(u-1):
            t = u - i - 1
            q_t1_u[:, t-1] = self.q_backward(q_t_t[:, t-1], q_t1_t[:, t-1], q_t1_u[:, t])


        log_l = np.sum(C)
        #log_l = 0
        A2 = self.batch_A(q_t_t, q_t1_t, q_t1_u)
        v2 = self.batch_v(q_t1_u, batch)

        pi2 = self.batch_pi(q_t1_u[:, 0])

        return A2, v2, pi2, log_l

    def log_lik(self, data, batches):
        self.batches = batches

        self.data = data
        
        log_l = 0
        for i, batch in enumerate(batches):
            self.n_batch = len(batch)
            self.c_batch = i
            users = len(batch)
            u = len(self.data[batch[0]])#data length
            q_t_t = np.zeros((users, u, self.hidden_states))
            q_t1_t = np.zeros((users, u-1, self.hidden_states))

            log_f = self.log_mubin(self.mubin_v, batch)

            time = 0
            tmp = self.pi
            tmp = np.tile(tmp, (users, 1))
            q_t_t[:, 0], C = self.q_update(log_f[:, time], tmp)

            for t in range(u-1):
                time = t + 1
                q_t1_t[:, t] = self.q_predict(q_t_t[:, t])
                q_t_t[:, t+1], C_t = self.q_update(log_f[:, time], q_t1_t[:, t])
                C += C_t
                
            log_l += np.sum(C)

        return log_l
    
    def prior_lik(self):
        l_lik = np.sum(np.nan_to_num((self.pi_prior - 1)*np.log(self.pi)))
        l_lik += np.sum(np.nan_to_num((self.v_prior - 1)*np.log(self.mubin_v)))
        l_lik += np.sum(np.nan_to_num((self.A_prior - 1)*np.log(self.A)))
        return l_lik

    def set_prior_single(self, A, v, pi, diag = None):
        self.A_prior = self.A_prior*0 + A
        if(not diag is None):
            for i in range(self.hidden_states):
                self.A_prior[i, i] = diag
        self.pi_prior = self.pi_prior*0 + pi
        self.v_prior = self.v_prior*0 + v

    def pred_newState(self, userid, items):
        if(userid in self.pred_user_dict):
            user_state, user_session = self.pred_user_dict[userid]
            if(len(items) > 0):
                user_session += items
                log_f = self.log_f_single(np.array(user_session))

                pred_user_state = self.q_update(log_f, user_state, axis = 0)

                self.pred_user_dict[userid] = [user_state, user_session]
            else:
                pred_user_state = user_state

        else:
            user_state = np.zeros(self.hidden_states)
            user_state = self.pi
            user_session = items
            self.pred_user_dict[userid] = [user_state, user_session]
            if(len(items) > 0):
                user_session += items
                log_f = self.log_f_single(np.array(user_session))

                pred_user_state = self.q_update(log_f, user_state, axis = 0)

                self.pred_user_dict[userid] = [user_state, user_session]
            else:
                pred_user_state = user_state

        return pred_user_state
            

    def pred_endSession(self, userid):
        if(userid in self.pred_user_dict):
            user_state, user_session = self.pred_user_dict[userid]
            
            log_f = self.log_f_single(np.array(user_session))

            pred_user_state = self.q_update(log_f, user_state, axis = 0)
            
            new_user_state = self.q_predict_single(pred_user_state, axis = 1)
            self.pred_user_dict[userid] = [new_user_state, []]
            return new_user_state

    def log_f_single(self, items):
        tmp = np.unique(items, return_counts=True)
        indx = tmp[0]
        num = tmp[1]

        return np.sum(np.log(self.mubin_v[:, indx])*num[None, :], axis = 1)

    def q_predict_single(self, q_current, axis = 1):
        """
        Take current state probability estimates and predict next probability according to A
        
        inputs:
        float[n_hidden_states] q_current: current state probability estimates

        returns:
        float[n_hidden_states] predicted state probability estimates
        """
        return np.sum(q_current*self.A.T, axis = axis)

    def relabel(self, true_v):
        n_states, n_items = true_v.shape

        if(n_items != self.mubin_v.shape[1]):
            print("Incompatible for relabeling: Item sizes, ", n_items, " and, ", self.mubin_v.shape[1])
        
        r = np.zeros(n_states, dtype = np.int32)

        for i in range(n_states):
            true_prob = true_v[i]
            best_p = -9e300
            best_j = -1
            for j in range(n_states):
                estimated_prob = self.mubin_v[j]
                p = np.sum(np.nan_to_num(np.log(estimated_prob)*true_prob))
                if p > best_p:
                    best_j = j
                    best_p = p
            r[i] = best_j
        
        print(r)
        #reorder
        pi_r = self.pi[r]
        v_r = self.mubin_v[r]
        A_r = self.A[r]
        A_r = A_r[:, r]

        #normalize
        pi_r = pi_r/sum(pi_r)
        v_r = self.row_norm(v_r)
        A_r = self.row_norm(A_r)
        
        return pi_r, A_r, v_r
