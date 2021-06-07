"""[Utils Library : Is divided into "GYM Data Preprocessing","RL Functions","Objective Functions",Printing Functions]"""
import numpy as np
import matplotlib.pyplot as plt
import torch 
import gym

"""[GYM Data Preprocessing]"""
def set_single_tensor(tensor):
	"""[Function that allows a single tensor to be processed by the neural networks that use Sequential function]

	Args:
		tensor ([type]): [single tensor whitout the batch dimention]

	Returns:
		tensor[type]: [returns a tensor with extra dimention so it can be prosseced by Sequential in pytorch]
	"""    
	tensor = tensor.unsqueeze(0)
	tensor = tensor.double() 
	return tensor

def format_data_type(x,data_type):
	"""[summary]

	Args:
		x ([type]): [description]
		data_type ([type]): [description]

	Returns:
		[type]: [description]
	"""    
	if  data_type=="Values":
		x=np.expand_dims(np.expand_dims(x, axis=0),0)
		#print("Values",x.shape)
		return "Values",x
	if data_type=="RGB_Image_RGB":
		x=np.expand_dims(x, axis=0)
		x=np.moveaxis(x, [0, 1, 2,3], [-4,-2, -1, -3])
		print("RGB_Image_RGB",x.shape)
		return "Images",x
	if data_type=="RGB_Image_GS":
		x = np.mean(x,2,keepdims = False)
		x = x[35:195]
		x = x[::2,::2]
		x=np.expand_dims(np.expand_dims(x, axis=0),0)
		print("RGB_Image_GS",x.shape)
		return  "Images",x

def format_data_type_tensor(x,data_type):
	"""[summary]

	Args:
		x ([type]): [description]
		data_type ([type]): [description]

	Returns:
		[type]: [description]
	"""    
	if  data_type=="Values":
		x=torch.unsqueeze(torch.unsqueeze(x,0),0)
		
	if data_type=="RGB_Image_RGB":
		x = torch.tensor([x], dtype=torch.float).to(self.inverse.model.device)
		x=x.permute(0,3,1,2)
		
	if data_type=="RGB_Image_GS":
		x = np.mean(x,2,keepdims = False)
		x = x[35:195]
		x = x[::2,::2]
		x = torch.tensor([x], dtype=torch.float).to(self.inverse.model.device)
		
	return x

def gym_spaces_dim(space):
	spece_dim=[]
	if isinstance( space, gym.spaces.Discrete):
		space_dim = space.n
	if isinstance( space, gym.spaces.Box):
		space_dim = space.shape
	return space_dim

"""[Printing Functions]"""
def plot_learning_curve(x, scores, figure_file):
	"""[summary]

	Args:
		x ([type]): [description]
		scores ([type]): [description]
		figure_file ([type]): [description]
	"""    
	running_avg = np.zeros(len(scores))
	for i in range(len(running_avg)):
		running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
	plt.plot(x, running_avg)
	plt.title('Running average of previous 100 scores')
	plt.savefig(figure_file)


"""[RL Functions]"""
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
