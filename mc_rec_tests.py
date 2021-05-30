#%%
import unittest
from mc_rec import MCRecommender
from mc_rec_sparse import MCRecommender as MCSparse
import numpy as np
import time
#%%

class MCRecTests(unittest.TestCase):
    tol = 1e-12
    rec = MCRecommender(3, 4, 1)
    def test_update(self):
        q_current = np.array([[0.6, 0.1, 0.3]])
        x = np.array([[0, 1, 0, 2]])
        v = np.array([[0.2, 0.1, 0.1, 0.6],
                      [0.1, 0.3, 0.2, 0.4],
                      [0.4, 0.2, 0.3, 0.1]])
        #calc with actual functions, not in logspace
        f = np.prod(v**x, axis = 1)
        q_hat = f*q_current
        q_true = q_hat/np.sum(q_hat)

        q_test = self.rec.q_update(q_current, x, v)
        
        self.assertTrue(np.max(np.abs(q_true - q_test)) < self.tol)

    def test_predict(self):
        A = np.array([
            [0.2, 0.2, 0.4, 0.2],
            [0.1, 0.4, 0.2, 0.3],
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.6, 0, 0]
        ])
        self.rec.A = A        

        q_simple = np.array([[1, 0, 0, 0]])

        q_test = self.rec.q_predict(q_simple)
        self.assertTrue(np.max(np.abs(A[0] - q_test[0])) < self.tol)

        q = np.array([[0.1, 0.3, 0.2, 0.4]])

        q_true = np.zeros(4)

        k = 4

        for i in range(k):
            q_true[i] = np.sum(A[:, i]*q)
        q_test = self.rec.q_predict(q)[0]

        self.assertTrue(np.max(np.abs(q_true - q_test)) < self.tol)




    def test_backward(self):
        A = np.array([
            [0.2, 0.2, 0.4, 0.2],
            [0.1, 0.4, 0.2, 0.3],
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.6, 0, 0]
        ])
        v = np.array([[0.2, 0.1, 0.1, 0.6],
                      [0.1, 0.3, 0.2, 0.4],
                      [0.4, 0.2, 0.3, 0.1],
                      [0.4, 0.2, 0.3, 0.1]])
        self.rec.A = A
        self.rec.mubin_v = v

        x = np.array([[0, 1, 0, 2]])

        q_t_t  = np.array([[0.1, 0.3, 0.2, 0.4]])
        q_t1_t  = self.rec.q_predict(q_t_t)
        q_t1_t  = self.rec.q_predict(q_t_t)
        q_t1_u = self.rec.q_update(q_t1_t, x, v)
        q_true = np.zeros(4)

        k = 4

        for i in range(k):
            q_true[i] = q_t_t[0,i]*np.sum(((A[i, :])/q_t1_t[0,:])*q_t1_u[0,:])


        q_test = self.rec.q_backward(q_t_t, q_t1_t, q_t1_u)
        self.assertTrue(np.max(np.abs(1 - np.sum(q_test))) < self.tol)
        self.assertTrue(np.max(np.abs(q_true - q_test)) < self.tol)


    def test_backprop(self):
        
        A = np.array([
            [1, 0],   #state 0
            [0.1, 0.9]#state 1
        ])

        pi = np.array([0.5, 0.5])

        v = np.array([
            [0.5, 0.499, 0.001], #state 0
            [0.33, 0.33, 0.34]        #state 1
        ])

        x = np.array([[
            [2, 2, 0],
            [1, 1, 0],
            [1, 2, 0],
            [1, 2, 0],
            [1, 2, 0],
            [1, 2, 0],
            [0, 0, 8]]#very unlikely if absorbed into state 0
        ])
        users = 200
        #x = self.generate_data(A, pi, v, users=users, length= 40)
        u = x.shape[1]#data length
        
        rec2 = MCRecommender(2, 3, 1)
        rec2.A = A
        rec2.mubin_v = v
        
        q_t_t = np.zeros((users, u, 2))
        q_t1_t = np.zeros((users, u-1, 2))
        q_t1_u = np.zeros((users, u, 2))


        tmp = pi
        tmp = np.tile(tmp, (users, 1))
        q_t_t[:, 0] = rec2.q_update(tmp, x[:,0], v)

        for t in range(u-1):
            q_t1_t[:, t] = rec2.q_predict(q_t_t[:, t])
            q_t_t[:, t+1] = rec2.q_update(q_t1_t[:, t], x[:, t+1], v)

        q_t1_u[:, u-1] = q_t_t[:, u-1]
        for i in range(u-1):
            t = u - i - 1
            q_t1_u[:, t-1] = rec2.q_backward(q_t_t[:, t-1], q_t1_t[:, t-1], q_t1_u[:, t])

        A2 = rec2.batch_A(q_t_t, q_t1_t, q_t1_u)
        print(A2/np.expand_dims(np.sum(A2, axis = 1), axis = -1))
        #Check final datapoint has back propogated
        self.assertTrue(q_t1_u[0, 0, 1] > 0.999)

    def test_stationary(self):
        tol = 1e-2
        
        A = np.array([
            [0.9, 0.1],   #state 0
            [0.1, 0.9]#state 1
        ])

        pi = np.array([0.5, 0.5])

        v = np.array([
            [0.5, 0.4, 0.1], #state 0
            [0.33, 0.33, 0.34]        #state 1
        ])
        users = 1000
        x = self.generate_data(A, pi, v, users=users, length= 100)
        u = x.shape[1]#data length
        
        rec2 = MCRecommender(2, 3, 1)
        rec2.A = A
        rec2.mubin_v = v
        
        q_t_t = np.zeros((users, u, 2))
        q_t1_t = np.zeros((users, u-1, 2))
        q_t1_u = np.zeros((users, u, 2))


        tmp = pi
        tmp = np.tile(tmp, (users, 1))
        q_t_t[:, 0] = rec2.q_update(tmp, x[:,0], v)

        for t in range(u-1):
            q_t1_t[:, t] = rec2.q_predict(q_t_t[:, t])
            q_t_t[:, t+1] = rec2.q_update(q_t1_t[:, t], x[:, t+1], v)

        q_t1_u[:, u-1] = q_t_t[:, u-1]
        for i in range(u-1):
            t = u - i - 1
            q_t1_u[:, t-1] = rec2.q_backward(q_t_t[:, t-1], q_t1_t[:, t-1], q_t1_u[:, t])

        A2 = rec2.batch_A(q_t_t, q_t1_t, q_t1_u)
        A2 = A2/np.expand_dims(np.sum(A2, axis = 1), axis = -1)

        v2 = rec2.batch_v(q_t1_u, x)
        v2 = v2/np.expand_dims(np.sum(v2, axis = 1), axis = -1)

        pi2 = rec2.batch_pi(q_t1_u[:, 0])
        pi2 = pi2/np.sum(pi2)
        print(A2)
        self.assertTrue(np.max(np.abs(A - A2)) < tol)

        self.assertTrue(np.max(np.abs(v - v2)) < tol)

        self.assertTrue(np.max(np.abs(pi - pi2)) < tol)

    def test_sparse(self):

        data = [[np.array([[0, 1], [2, 1]])
                ,np.array([[0, 2], [1, 3]])
                ,np.array([[0, 1, 2], [1, 2, 4]])],
                [np.array([[1], [2]])
                ,np.array([[0, 1], [1, 5]])]]
        v = np.array([[0.1, 0.5, 0.4], [0, 1, 0], [0.1, 0.8, 0.1]])

        sparse = MCSparse(v.shape[0], v.shape[1], 2)
        sparse.data = data
        sparse.time = 0

        batch = [1]
        sparse.n_batch = len(batch)
        x_test = sparse.log_mubin(v, batch)
        self.assertTrue(x_test[1] == 0)


        #float[n_batch * n_timesteps * n_hidden_states] Q_t1_s
        Q_t1_s = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        sparse.batch_v(Q_t1_s, [0])

        Q_t1_s = np.array([[[1, 0, 0], [0, 1, 0]]])
        sparse.batch_v(Q_t1_s, [1])

        


        
    def generate_data(self, A, pi, v, users = 1000, length = 30):
        K = v.shape[0]
        N = v.shape[1]
        
        states = np.array(range(K))
        items = np.array(range(N))
        sizes = np.array(range(20))+1

        x = np.zeros([users, length, N], dtype = np.int8)
        session = np.zeros(N, dtype = np.int8)

        for u in range(users):
            current_state = np.random.choice(states, p = pi)
            for t in range(length):
                session *= 0
                s = np.random.choice(sizes)
                session_i = np.random.choice(items, s, p = v[current_state])
                np.add.at(session, session_i, 1)
                x[u, t] = session

                current_state = np.random.choice(states, p = A[current_state])
        
        return x
    
    def generate_data_sparse(self, A, pi, v, users = 1000):
        K = v.shape[0]
        N = v.shape[1]
        
        states = np.array(range(K))
        items = np.array(range(N))
        sizes = np.array(range(20))+1

        lengths = np.array(range(20))+10

        data = []
        user_data = []

        for u in range(users):
            current_state = np.random.choice(states, p = pi)
            length = np.random.choice(lengths)
            for t in range(length):
                s = np.random.choice(sizes)
                session_i = np.random.choice(items, s, p = v[current_state])
                
                session = np.array(np.unique(session_i, return_counts= True))
                
                user_data.append(session)

                current_state = np.random.choice(states, p = A[current_state])
        
            data.append(user_data)
            user_data = []
        return data

    def generate_data_both(self, A, pi, v, users = 1000):
        K = v.shape[0]
        N = v.shape[1]
        
        states = np.array(range(K))
        items = np.array(range(N))
        sizes = np.array(range(20))+1
        lengths = np.array(range(20))+10

        data = []
        user_data = []
        
        session = np.zeros(N, dtype = np.int8)
        user_sessions = []

        for u in range(users):
            current_state = np.random.choice(states, p = pi)
            length = np.random.choice(lengths)
            for t in range(length):
                s = np.random.choice(sizes)
                
                session_i = np.random.choice(items, s, p = v[current_state])
                
                sessions.append(session_i)

                session = np.array(np.unique(session_i, return_counts= True))
                
                user_data.append(session)

                current_state = np.random.choice(states, p = A[current_state])
        
            data.append(user_data)
            user_data = []
            user_sessions.append(sessions)
        return data, user_sessions

    def generate_data_full(self, A, pi, v, users = 1000):
        K = v.shape[0]
        N = v.shape[1]
        
        states = np.array(range(K))
        items = np.array(range(N))
        sizes = np.array(range(20))+1

        lengths = np.array(range(10))+5

        #x = np.zeros([users, length, N], dtype = np.int8)
        session = np.zeros(N, dtype = np.int8)
        user_sessions = []
        for u in range(users):
            current_state = np.random.choice(states, p = pi)
            length = np.random.choice(lengths)
            sessions = []
            for t in range(length):
                
                s = np.random.choice(sizes)
                session_i = np.random.choice(items, s, p = v[current_state])
                sessions.append(session_i)
                current_state = np.random.choice(states, p = A[current_state])
            user_sessions.append(sessions)
        return user_sessions

"""
#vectorization Speedup test
#nested lists W.O. vectorization
#faster than(for large datasets)
#index arrays
#%%
#list add test
items = 10
sessions = 10
users = 5000

#gen normal list
data_list = []
for i in range(users):
    user_dat = []
    for j in range(sessions):
        session_i = []
        session_n = []
        for k in range(items):
            session_i.append(0)
            session_n.append(1)

        user_dat.append([np.array(session_i), np.array(session_n)])
        session = []
    data_list.append(user_dat)
    user_dat = []

#gen index list
dims = 4
enteries = items*users*sessions
dat_vec = np.zeros((dims,enteries), dtype=np.int32)
e = 0
for i in range(users):
    for j in range(sessions):
        for k in range(items):
            dat_vec[0, e] = i
            dat_vec[1, e] = j
            dat_vec[2, e] = 0#item index
            dat_vec[3, e] = 1#item num
            e += 1


#%%
classes = 500
items = 100
p_s = np.zeros((users, sessions, classes))
v = np.zeros((classes, items)) + 1.0/items
start = time.time()
for i in range(users):
    for j in range(sessions):
        u_dat =data_list[i][j] 
        indx = u_dat[0]
        num = u_dat[1]
        p_s[i, j] = np.sum(np.log(v[:, indx])*num[None, :], axis = 1)
end = time.time()

print("List: ", end-start)
p_s = np.zeros((users, sessions, classes))

start = time.time()
np.add.at(p_s, [dat_vec[0], dat_vec[1]], (np.log(v[:, dat_vec[2]])*dat_vec[3][None, :]).T)
end = time.time()

print("vec: ", end-start)
"""
#%%
if __name__ == "__main__":
    unittest.main()
