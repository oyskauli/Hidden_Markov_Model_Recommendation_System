#%%
import torch
print(torch.cuda.is_available())
import numpy as np
from mc_data_func import mc_import_data, mc_batch_data, mc_import_items, sync_user_item_data, categorical_var, generate_dist_mc, generate_data
from mc_rec_sparse import MCRecommender, fit_batch, fit_batch_cuda, fit_batch_cuda_seq, log_lik_cuda, state_popularity_cuda_seq, log_mubin_v, batch_v_v, log_mubin_v_torch,batch_v_v_torch
import time
from joblib import Parallel, delayed, parallel_backend
from typing import *
import os
from PYTORCHHelper import ItemsModule, map_itemids, filter_unks
import pickle
import copy
c_path = #Global path to current dir

#%%
def optimizer_to(optim, device):
    """
    Move optimizer parameters, to save GPU mem
    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
#%%

def log_mubin_v_torch(v, batch, n_batch, data):
    """
    get probability of click within each class using torch
    """
    return torch.transpose(torch.transpose(torch.log(v[:,batch[:]]), 0, 1), 1, 2)

def batch_v_v_torch(Q_t1_s, batch, data, v_shape, Vs, dtype = torch.float64):
    """
    Generate result for V within batch using torch
    """
    b = torch.flatten(batch)
    q = torch.transpose(torch.flatten(Q_t1_s, 0, 1), 0, 1)

    Vs.index_add_(1, b, q)

def batch_A_torch(Q_t_t, Q_t1_t, Q_t1_s, A, As=None, dtype = np.float64):
    """
    Unnormalized transition probabilities, for batch of users using torch

    inputs:
    float[n_batch * n_timesteps   * n_hidden_states] Q_t_t:  probability estimates at t given data up to t
    float[n_batch * n_timesteps-1 * n_hidden_states] Q_t1_t: probability estimates at t + 1 given data up to t
    float[n_batch * n_timesteps   * n_hidden_states] Q_t1_s: probability estimates at t + 1 given all data

    returns:
    float[n_hidden_states * n_hidden_states] Unnormalized transition probability estimates for batch
    """
    n_t = Q_t_t.shape[1]
    hidden_states = A.shape[0]
    if(As is None):
        As = torch.zeros((hidden_states, hidden_states), requires_grad=False, device=torch.device("cuda"), dtype = torch.float)

    for i in range(n_t - 1):
        As += torch.sum((A/torch.unsqueeze(Q_t1_t[:, i], axis = 1))*torch.unsqueeze(Q_t_t[:, i], axis = -1)*torch.unsqueeze(Q_t1_s[:, i+1], axis = 1) ,axis = 0)     
    return As

#%%
def _gpu_only_fit_(PIs, As, Vs, A, mubin_v, pi, batches, data, hidden_states, device, dtype = torch.float64):
    """
    Fit function to be used in Joblib paralell, higher gpu mem usage
    """
    As = torch.tensor(As, requires_grad=False, dtype = dtype).to(device)
    Vs = torch.tensor(Vs, requires_grad=False, dtype = dtype).to(device)
    PIs = torch.tensor(PIs, requires_grad=False, dtype = dtype).to(device)

    A = torch.tensor(A, requires_grad=False).to(device)
    mubin_v = torch.tensor(mubin_v, requires_grad=False).to(device)
    pi = torch.tensor(pi, requires_grad=False).to(device)
    log_l = 0
    for j, b in enumerate(batches):
        b = torch.tensor(b, requires_grad=False, dtype=torch.long).to(device)
        n_batch = len(b)
        c_batch = j
        
        l = fit_batch_cuda_seq(b, data, hidden_states, A, pi, mubin_v, PIs, As, Vs, f_mubin = log_mubin_v_torch, batchv=batch_v_v_torch, dtype=dtype)
        log_l += l
        if((j+1)%20 == 0):
            print(j+1, "/", len(batches))
        del b

    As = As.cpu().numpy()
    Vs = Vs.cpu().numpy()
    PIs = PIs.cpu().numpy()
    return As, Vs, PIs, log_l



class factorizationModel(torch.nn.Module):
    """
    Model to generate a matrix factorization of the observed 
    state-item clicks matrix
    """
    def __init__(self, K, N, embed_dim, cat_vars, device, penalty = 0, dtype=torch.float64):
        super(factorizationModel, self).__init__()
        self.penalty = penalty
        self.prior = 0
        self.dtype=dtype

        #state embeddings
        S = torch.empty(K, embed_dim, device=device, dtype=self.dtype)
        
        #item embeddings
        B = torch.empty(N, embed_dim, device=device, dtype=self.dtype)
        B_intercept = torch.empty(N, device=device, dtype=self.dtype)
        
        #Create parameters for the given categorical variables
        #cat vars to be given as indexes for each item
        self.cat_param = []
        self.cat_vals = []
        i = 0
        for v in cat_vars:
            cp = torch.empty(K, np.max(v) + 1, device=device, dtype=self.dtype)
            torch.nn.init.xavier_uniform(cp, 0.1)
            self.cat_param.append(torch.nn.Parameter(cp))
            self.register_parameter("cat_par" + str(i), self.cat_param[-1])
            self.cat_vals.append(torch.tensor(v, device=device, dtype=torch.long, requires_grad=False))
            i += 1
        self.S = torch.nn.Parameter(S)
        self.B = torch.nn.Parameter(B)
        self.B_int = torch.nn.Parameter(B_intercept)

        torch.nn.init.xavier_uniform(self.S, 0.1)
        torch.nn.init.xavier_uniform(self.B, 0.1)
    
    def forward(self, X):
        #transform to wanted form(can be removed)
        S, B, B_int = self.restrict()
        #calculate likelihood
        log_lik = self.factor_lik_torch(X, S, B, B_int)

        return log_lik + self.get_penalty(S, self.penalty)

    def restrict(self):
        #Requiering S matrix to determine the std 
        std = torch.std(self.B, dim = 0)
        B = self.B/std
        S = self.S*std

        #centering, df that will be normalized away
        B = B - torch.mean(B, axis = 0)
        #Removing implicit intercept
        col_sum = torch.mean(S, axis = 0)
        S = S - col_sum
        
        B_int = self.B_int + col_sum @ B.T

        return S, B, B_int
        

    def mubin(self):
        return self._mubin_(self.S, self.B, self.B_int)

    def _mubin_(self, S, B, B_int):
        """
        Calculate linear combination of factorization
        """
        P = S @ B.T

        P += B_int

        for i in range(len(self.cat_param)):
            c = self.cat_param[i] - torch.unsqueeze(torch.mean(self.cat_param[i], axis=1), axis = -1)
            P += c[:, self.cat_vals[i]]

        return P

    def generate_mubin_LR_torch(self):
        """
        float[states, embedding_dim] S
        float[items, embedding_dim] B

        Takes state and item embeddings and 
        generates item selection probabilities.
        With a Logit link
        """
        P = self.mubin()
        #Norm
        P -= torch.max(P, axis = 1).values.unsqueeze(1)
        P = P - torch.log(torch.sum(torch.exp(P), axis = 1).unsqueeze(1))

        return P

    def get_selection_prob(self):
        return self._get_selection_prob_(self.S, self.B, self.B_int)
    
    def _get_selection_prob_(self, S, B, B_int):
        """
        Calculate log of selection probability from factorization
        """
        P = self._mubin_(S, B, B_int)
        #Norm
        P -= torch.max(P, axis = 1).values.unsqueeze(1)
        P = P - torch.log(torch.sum(torch.exp(P), axis = 1).unsqueeze(1))

        return(P)

    def factor_lik_torch(self, X, S, B, B_int):
        """
        Calculate log-likelihood of model
        """
        X = X + self.prior/self.S.shape[0]#item prior remove if needed
        P = self._get_selection_prob_(S, B, B_int)
        L = -torch.sum(P*X)# + 0.001*torch.sum(self.S**2) + 0.001*torch.sum(self.B**2)
        return L

    def get_penalty(self, S, lam):
        return lam*torch.sum(S**2)

    def move(self, optimizer, device):
        optimizer_to(optimizer, device)
        self.to(device)


class predictionModel(ItemsModule):
    def __init__(self, pi, A, V, item_dat):
        super(predictionModel, self).__init__()
        self.pi = torch.as_tensor(pi, dtype=torch.float32)
        self.A = torch.as_tensor(A, dtype=torch.float32)
        self.V = torch.as_tensor(V, dtype=torch.float32)
        
        self.log_pi = torch.as_tensor(np.log(pi), dtype=torch.float32)
        #self.log_A = torch.as_tensor(np.log(A), dtype=torch.float32)
        self.log_V = torch.as_tensor(np.log(V), dtype=torch.float32)
        
        self.output_lookup = dict(zip(item_dat["id"], item_dat.index))
        self.q = torch.clone(self.pi)

    def __forward_predict_tmp__(self, x, targets: Optional[torch.Tensor]=None):
        
        #initial intrest propalilities
        q = torch.clone(self.pi)

        #probability of selected items for each intrest group
        log_f = torch.log(self.V[:, x])

        for i in range(len(x)):
            #update to data
            q = torch.log(q) + log_f[:, i]
            
            #normalize
            q -= torch.max(q)
            q = torch.exp(q)
            #q = q/torch.sum(q)
            
            #predict
            q = torch.sum(q*self.A.T, dim = 1)
        
        top_n = q.argsort()
        top_n = top_n[-10:]
        #only do weighted sum over top-10 states for faster compute time
        #will almost always give same result in top-n items
        if targets is None:
            scores = torch.sum(torch.unsqueeze(q[top_n], 1)*self.V[top_n], dim=0)
        else:
            scores = torch.sum(torch.unsqueeze(q[top_n], 1)*self.V[:, targets][top_n, :], dim=0)


        return scores

    def __forward_predict__(self, x, targets: Optional[torch.Tensor]=None):
        
        #initial intrest propalilities
        q = torch.clone(self.pi)

        #probability of selected items for each intrest group
        log_f = self.log_V[:, x]

        for i in range(len(x)):
            #update to data
            q = torch.log(q) + log_f[:, i]
            
            #normalize
            q -= torch.max(q)
            q = torch.exp(q)
            #q = q/torch.sum(q)
            
            #predict
            q = torch.sum(q*self.A.T, dim = 1)
        
        q = q/torch.sum(q)

        top_n = q.argsort()
        top_n = top_n[-10:]
        #only do weighted sum over top-10 states for faster compute time
        #will almost always give same result in top-n items
        if targets is None:
            scores = torch.sum(torch.unsqueeze(q[top_n], 1)*self.V[top_n], dim=0)
        else:
            scores = torch.sum(torch.unsqueeze(q[top_n], 1)*self.V[:, targets][top_n, :], dim=0)


        return scores

    def __forward_prank__(self, data: Dict[str, List[str]], targets: torch.Tensor) -> Optional[torch.Tensor]:
        
        item_input = data['userItemHistory'] if 'userItemHistory' in data else data['itemId']
        x = filter_unks(map_itemids(item_input, self.output_lookup), -1)

        if len(x) == 0:  # Check item validity to early escape
            return None

        #initial intrest propalilities
        q = torch.clone(self.pi)

        #probability of selected items for each intrest group
        log_f = self.log_V[:, x]

        for i in range(len(x)):
            #update to data
            q = torch.log(q) + log_f[:, i]
            
            #normalize
            q -= torch.max(q)
            q = torch.exp(q)
            #q = q/torch.sum(q)
            
            #predict
            q = torch.sum(q*self.A.T, dim = 1)
        
        q = q/torch.sum(q)

        top_n = q.argsort()
        top_n = top_n[-10:]
        #only do weighted sum over top-10 states for faster compute time
        #will almost always give same result in top-n items
        if targets is None:
            scores = torch.sum(torch.unsqueeze(q[top_n], 1)*self.V[top_n], dim=0)
        else:
            scores = torch.sum(torch.unsqueeze(q[top_n], 1)*self.V[:, targets][top_n, :], dim=0)


        return scores
#%%

class MCRecommender_F(MCRecommender):
    def __init__(self, hidden_states, embedding_dim, items, users, cat_vars = [], dtype = np.float64):
        super().__init__(hidden_states, items, users, dtype)
        if(dtype== np.float64):
            dtype_t=torch.float64
        elif(dtype== np.float32):
            dtype_t=torch.float32
        else:
            print("dtype not recognized, using torch.float64 as default")
            dtype_t=torch.float64
        self.dtype_t = dtype_t

        self.embedding_dim = embedding_dim
        self.initialized_embeddings = False
        self.cat_vars = cat_vars
        self.max_iter = 500
        self.conv_hist = None

        if(torch.cuda.is_available()):
            self.device = torch.device("cuda")
            print("cuda")
        else: 
            self.device = torch.device("cpu")
            print("cpu")
        self.model = factorizationModel(self.hidden_states, self.items, self.embedding_dim, self.cat_vars, self.device, dtype=dtype_t)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.04)
        self.model.move(self.optimizer, "cpu")
        torch.cuda.empty_cache()
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
                self.n_batch = len(b)
                self.c_batch = j
                A2, v2, pi2, l = self.fit_batch(b)
                log_l += l
                As += A2
                Vs += v2
                PIs += pi2
                #print("current lik: ", log_l)
            if(i%100 == 0):
                print("Batch: ", i, "/", len(batches))
            new_A = self.row_norm(As + self.A_prior - 1)
            #print("Diff:", np.max(np.abs(self.A-new_A)))
            self.A = new_A
            self.fit_factorization(Vs)
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
    
    def fit_parallel(self, data, batches, max_iter = 200, tol = 1, threads = 10, low_gpu_mem = True):
        self.batches = batches

        self.data = data

        converged = False
        good_updates = 0
        slow_updates = 0
        log_l_prev = 1

        conv_hist = []
        
        As = np.zeros([self.hidden_states, self.hidden_states], dtype=self.dtype)
        Vs = np.zeros([self.hidden_states, self.items], dtype=self.dtype)
        PIs = np.zeros([self.hidden_states], dtype=self.dtype)
        for i in range(max_iter):
            print("Iteration: ", i)
            log_l = 0
            i_s = 0
            if(threads > len(batches)):
                i_e = len(batches)
            else:
                i_e = threads
            
            hidden_states = self.hidden_states
            A = self.A
            pi = self.pi
            mubin_v = self.mubin_v
            if(low_gpu_mem):
                with parallel_backend("threading", n_jobs = threads):
                    while (i_s < len(batches)):
                        batch_batch = batches[i_s:i_e]
                        #print(type(batch_batch))
                        print("Starting Batch: ", i_s, "-", i_e)
                        #res = fit_batch(batch_batch[0], data, hidden_states, A, pi, mubin_v, f_mubin = log_mubin_v, batchv=batch_v_v)
                        res = Parallel()(delayed(fit_batch_cuda)(b, data, hidden_states, A, pi, mubin_v, f_mubin = log_mubin_v, batchv=batch_v_v) for b in batch_batch)
                        print("Batch Done")
                        for r in res:
                            A2, v2, pi2, l = r
                            log_l += l
                            As += A2
                            Vs += v2
                            PIs += pi2
                        i_s = i_e
                        print(Vs)
                        if(i_e+threads > len(batches)):
                            i_e = len(batches)
                        else:
                            i_e = i_e+threads
                        print(i_e, "/", len(batches))
                        #print("current lik: ", log_l)
            else:
                batches_s = np.array_split(batches, threads)
                with parallel_backend("threading", n_jobs = threads):
                    res = Parallel()(delayed(_gpu_only_fit_)(PIs, As, Vs, A, mubin_v, pi, b, 0, self.hidden_states, self.device, dtype=self.dtype_t) for b in batches_s)
                    log_l = 0
                    for r in res:
                        A2, v2, pi2, l = r
                        log_l += l
                        As += A2
                        Vs += v2
                        PIs += pi2

              
            new_A = self.row_norm(As + self.A_prior - 1)
            #print("Diff:", np.max(np.abs(self.A-new_A)))
            self.A = new_A
            self.fit_factorization(Vs, max_iter = self.max_iter)
            self.pi = self.row_norm(PIs + self.pi_prior -1)
            As = As*0
            Vs = Vs*0
            PIs = PIs*0
            print("Total lik: ", log_l)
            if(log_l_prev < 0 and tol > 0):#not first iter
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
            conv_hist.append(log_l)
        self.conv_hist = conv_hist
        return log_l, self.prior_lik()

    def get_parameters(self, data, batches, threads = 10, low_gpu_mem = True):
        self.batches = batches

        As = np.zeros([self.hidden_states, self.hidden_states], dtype=self.dtype)
        Vs = np.zeros([self.hidden_states, self.items], dtype=self.dtype)
        PIs = np.zeros([self.hidden_states], dtype=self.dtype)
        batches_s = np.array_split(batches, threads)

        hidden_states = self.hidden_states
        A = self.A
        pi = self.pi
        mubin_v = self.mubin_v

        with parallel_backend("threading", n_jobs = threads):
            res = Parallel()(delayed(_gpu_only_fit_)(PIs, As, Vs, A, mubin_v, pi, b, 0, self.hidden_states, self.device, dtype=self.dtype_t) for b in batches_s)
            log_l = 0
            for r in res:
                A2, v2, pi2, l = r
                log_l += l
                As += A2
                Vs += v2
                PIs += pi2

        A = self.row_norm(As + self.A_prior - 1)
        #print("Diff:", np.max(np.abs(self.A-new_A)))
        Vs = self.row_norm(Vs + self.v_prior -1)
        PIs = self.row_norm(PIs + self.pi_prior -1)

        return PIs, A, Vs

    

    def log_lik_parallel(self, data, batches, threads = 10):
        self.batches = batches

        self.data = data

        log_l = 0
        i_s = 0
        if(threads > len(batches)):
            i_e = len(batches)
        else:
            i_e = threads
        
        hidden_states = self.hidden_states
        A = self.A
        pi = self.pi
        mubin_v = self.mubin_v
        with parallel_backend("threading", n_jobs = threads):
            while (i_s < len(batches)):
                batch_batch = batches[i_s:i_e]
                #print(type(batch_batch))
                print("Starting Batch: ", i_s, "-", i_e)
                #res = fit_batch(batch_batch[0], data, hidden_states, A, pi, mubin_v, f_mubin = log_mubin_v, batchv=batch_v_v)
                res = Parallel()(delayed(log_lik_cuda)(b, data, hidden_states, A, pi, mubin_v, f_mubin = log_mubin_v, batchv=batch_v_v) for b in batch_batch)
                print("Batch Done")
                for r in res:
                    log_l += r
                i_s = i_e
                if(i_e+threads > len(batches)):
                    i_e = len(batches)
                else:
                    i_e = i_e+threads
                print(i_e, "/", len(batches))
                #print("current lik: ", log_l)
                
        return log_l, self.prior_lik()

    def state_pop_parallel(self, data, batches, threads = 10):
        self.batches = batches

        self.data = data

        log_l = 0
    
        hidden_states = self.hidden_states
        A = self.A
        pi = self.pi
        mubin_v = self.mubin_v
        
        A = torch.tensor(A, requires_grad=False).to(self.device)
        mubin_v = torch.tensor(mubin_v, requires_grad=False).to(self.device)
        pi = torch.tensor(pi, requires_grad=False).to(self.device)


        Corr = np.zeros([self.hidden_states, self.hidden_states], dtype=self.dtype)
        POP = np.zeros([self.hidden_states], dtype=self.dtype)
        VAR = np.zeros([self.hidden_states], dtype=self.dtype)
        
        log_l = 0
        for j, b in enumerate(batches):
            b = torch.tensor(b, requires_grad=False, dtype=torch.long).to(self.device)
            n_batch = len(b)
            c_batch = j
            
            p, c, v = state_popularity_cuda_seq(b, data, hidden_states, A, pi, mubin_v, f_mubin = log_mubin_v_torch, batchv=batch_v_v_torch, dtype=self.dtype_t)
            POP += p
            Corr += c
            VAR += v
            if((j+1)%20 == 0):
                print(j+1, "/", len(batches))
            del b

        N = 0
        t= 0
        for b in batches:
            N+=np.prod(b.shape)

        POP = POP/N
        VAR = VAR/N - (POP)**2
        SD = np.sqrt(VAR)

        for i in range(self.hidden_states):
            for j in range(i, self.hidden_states):
                Corr[i, j] = Corr[i, j]/N - POP[i]*POP[j]
                Corr[i, j] = Corr[i, j]/(SD[i]/SD[j])
                    
        return POP, Corr

    def initialize(self):
        """
        sample initial values according to prior

        float add_p, added to prior, larger => larger inital values 
        """

        self.A = self.row_init(self.A, self.A_prior) 

        self.mubin_v = self.row_init(self.mubin_v, self.v_prior)

        self.pi = self.row_init(self.pi, self.pi_prior)
    

    def get_fact_mubin(self, model = None):
        if(model is None):
            model = self.model
        v = np.exp(np.array(model.get_selection_prob().detach().cpu()))
        torch.cuda.empty_cache()
        return v

    def fit_factorization(self, Vs, tol = 0.01, max_iter = 500):
        #n_k = np.sum(Vs, axis = 1)
        
        print("start")
        self.model.move(self.optimizer, self.device)
        Xt = torch.tensor(Vs, device = self.device)
        for i in range(max_iter):
            self.optimizer.zero_grad()
            loss = self.model(Xt)
            loss.backward()

            self.optimizer.step()
            #print(float(loss))
        print(Xt)
        Xt = Xt.detach().cpu()
        del Xt

        self.mubin_v = self.get_fact_mubin()
        self.model.move(self.optimizer, "cpu")
        torch.cuda.empty_cache()
        print(self.mubin_v)

    def save(self, filename):
        f = open(filename, "wb")
        model_params = [self.A, self.pi, self.A_prior, self.pi_prior, self.conv_hist]
        pickle.dump(model_params, f)

        split = filename.split(".")
        torchpath = ".".join(split[0:-1]) +"_torch_model." + split[-1]
        torch.save(self.model.state_dict(), torchpath)

    def load(self, filename, load_torch = True):
        f = open(filename, "rb")
        model_params = pickle.load(f)
        if(len(model_params) == 4):
            self.A, self.pi, self.A_prior, self.pi_prior = model_params
        else:
            self.A, self.pi, self.A_prior, self.pi_prior, self.conv_hist = model_params

        split = filename.split(".")
        torchpath = ".".join(split[0:-1]) +"_torch_model." + split[-1]
        
        if(load_torch):
            split = filename.split(".")
            torchpath = ".".join(split[0:-1]) +"_torch_model." + split[-1]
            self.model.load_state_dict(torch.load(torchpath))
            self.mubin_v = self.get_fact_mubin()
        else:
            embed_dim = 50
            model2 = factorizationModel(self.hidden_states, self.items, embed_dim, self.cat_vars, self.device, dtype=self.dtype_t)

            model2.load_state_dict(torch.load(torchpath))
            self.mubin_v = self.get_fact_mubin(model2)
            del model2

    def pred_newState(self, userid, item = None):
        if(userid in self.pred_user_dict):
            user_state, user_session = self.pred_user_dict[userid]
        else:
            user_state = np.zeros(self.hidden_states)
            user_state[:] = self.pi
            if(item is None):
                user_session = []
            else:
                user_session = [item]
            self.pred_user_dict[userid] = [user_state, user_session]
        
        if(item is None):
            pred_user_state = user_state
            pred_user_state = self.q_predict_single(pred_user_state)
        else:
            user_session.append(item)
            log_f = np.log(self.mubin_v[:, item])
            pred_user_state = self.q_update(log_f, user_state, axis = 0)[0]
            pred_user_state = np.sum(pred_user_state*self.A.T, axis = 1)
            self.pred_user_dict[userid] = [pred_user_state, user_session]


        return pred_user_state
    
    def cast_to(self, np_dtype, torch_dtype):
        self.pi = self.pi.astype(np_dtype)
        self.A = self.A.astype(np_dtype)
        self.mubin_v = self.mubin_v.astype(np_dtype)
        self.dtype = np_dtype
        self.dtype_t = torch_dtype

    def reclass(self, new_states, POP, CORR):
        if(new_states > self.hidden_states):
            print("This funtion is only capable of collapsing classes")
        c_states = self.hidden_states
        while(new_states < c_states):
            max_corr = 0
            pos = []

            for i in range(c_states):
                for j in range(i+1, c_states):
                    if(CORR[i, j] > max_corr):
                        max_corr = CORR[i, j]
                        pos = [i, j]
            i, j = pos
            self.pi[i] = self.pi[i] + self.pi[j]

            self.mubin_v[i] = (self.mubin_v[i]+self.mubin_v[j])/2

            self.A[:, i] = self.A[:, i] + self.A[:, j]

            self.A[i] = (self.A[i] + self.A[j])/2

            CORR[i] = (CORR[i]+CORR[j])/2
            CORR[:, i] = (CORR[:, i]+CORR[:, j])/2
            

            r = np.arange(c_states)
            r = np.delete(r, j)
            #reorder
            self.pi = self.pi[r]
            self.mubin_v = self.mubin_v[r]
            self.A = self.A[r]
            self.A = self.A[:, r]
            CORR = CORR[r]
            CORR = CORR[:, r]
            
            self.pi_prior = self.pi_prior[r]
            self.v_prior = self.v_prior[r]
            self.A_prior = self.A_prior[r]
            self.A_prior = self.A_prior[:, r]

            c_states = c_states - 1

        self.hidden_states = c_states
        self.model = factorizationModel(self.hidden_states, self.items, self.embedding_dim, self.cat_vars, self.device, dtype=self.dtype_t)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.04)
        self.model.move(self.optimizer, "cpu")
        torch.cuda.empty_cache()
