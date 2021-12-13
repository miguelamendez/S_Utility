"""[RL Functions]"""
import numpy as np
import torch
#def vectorized(function):
#    def vector_wrapper(*args,**vargs):
#        num_workers:
#        vecor_arr=[]
#        for i in num_workers:
#        vector_arr.append(function(),i)
#        return vector_
        

def advantage_gen(reward_batch,value_batch,gamma,gae_lambda):
	advantage = np.zeros(len(reward_arr), dtype=np.float32)
	for t in range(len(reward_arr)-1):
		discount = 1
		a_t = 0
		for k in range(t, len(reward_arr)-1):
			a_t += discount*(reward_arr[k] + gamma*values[k+1]*\
				(1-int(dones_arr[k])) - values[k])
			discount *= gamma*gae_lambda
		advantage[t] = a_t
		advantage = torch.tensor(advantage).to(self.device)

def make_train_data(reward, done, value, gamma, num_step):
	discounted_return = np.empty([num_step])
	# Discounted Return
	if use_gae:
		gae = np.zeros_like([num_worker, ])
		for t in range(num_step - 1, -1, -1):
			delta = reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]
			gae = delta + gamma * lam * (1 - done[:, t]) * gae
			discounted_return[:, t] = gae + value[:, t]
	# For Actor
		adv = discounted_return - value[:, :-1]
	else:
		running_add = value[:, -1]
		for t in range(num_step - 1, -1, -1):
			running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
			discounted_return[:, t] = running_add
	# For Actor
		adv = discounted_return - value[:, :-1]
	return discounted_return.reshape([-1]), adv.reshape([-1])

