import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import optimize
import seaborn as sns
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
'''
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
'''
import pandas as pd


ending = "012523"



def plot_ae_loss(train_loss, val_loss):

	plt.plot(train_loss, label="Train Loss")
	plt.plot(val_loss, label="Val Loss")
	plt.title("AE Loss")
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(loc='upper right')
	#plt.show()
	plt.savefig("plots/loss_%s.png"%(ending))
	plt.close()
'''
class AE(torch.nn.Module):
	def __init__(self):
		super().__init__()

		# ENCODER
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(2100, 1024),
			torch.nn.ReLU(),
			torch.nn.Linear(1024, 512),
			torch.nn.ReLU(),
			torch.nn.Linear(512, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 16)
			)

		# DECODER
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(16, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 512),
			torch.nn.ReLU(),
			torch.nn.Linear(512, 1024),
			torch.nn.ReLU(),
			torch.nn.Linear(1024, 2100)
		
			)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

'''
def plot_heatmap(input, output, label):
	
	input = input.reshape(700,3)
	output = output.reshape(700,3)
		
	input = pd.DataFrame(input, columns = ["pT","eta","phi"])
	input = input.loc[~(input==0).all(axis=1)]
	output = pd.DataFrame(output, columns = ["pT","eta","phi"])
	output = output.loc[~(output==0).all(axis=1)]
	#print("============ INPUT ==============")
	#print(input)
	#print("============ OUTPUT ==============")
	#print(output)
	'''
	input_map = input.pivot_table(index="eta",columns="phi",values="pT",aggfunc='mean')
	print(input_map)
	sns.heatmap(input_map)	
	plt.savefig("input_heatmap_%s.png"%label)
	plt.close()
	output_map = output.pivot_table(index="pT",columns="phi",values="eta",aggfunc='mean')
	sns.heatmap(output_map)	
	plt.savefig("output_heatmap_%s.png"%label)
	plt.close()
	'''
	fig = plt.figure(figsize=(10,10))
	ax = fig.gca(projection='3d')
	#ax.set_box_aspect(aspect = 3)
	#surf = ax.plot_trisurf(input['eta'],input['phi'],input['pT'], cmap=plt.cm.coolwarm, linewidth=0.4)
	surf = ax.bar3d(input['eta'],input['phi'],5,0.2,0.2,input['pT'],color='g',shade=True)
	in_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")
	ax.set_xlim(-4.5,4.5)
	ax.set_ylim(-3.2,3.2)
	ax.set_xlabel("eta")
	ax.set_ylabel("phi")
	ax.set_zlabel("pT")
	#ax.set_zlim(0,400)
	#ax.set_box_aspect(aspect = 3)
	#fig.colorbar(surf, shrink=0.5, aspect=5)
	#plt.savefig("plots/input_heatmap_%s.png"%label)
	#plt.close()
	#fig = plt.figure(figsize=(20,20))
	#ax = fig.gca(projection='3d')
	#ax.set_box_aspect(aspect = 3)
	#surf = ax.plot_trisurf(output['eta'],output['phi'],output['pT'], cmap=plt.cm.coolwarm, linewidth=0.4)
	surf2 = ax.bar3d(output['eta'],output['phi'],5,0.2,0.2,output['pT'],shade=True)
	out_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
	ax.legend([in_proxy,out_proxy],['input event','AE output event'])
	#ax.set_xlim(-4.,4.)
	#ax.set_ylim(-3.,3.)
	#ax.set_zlim(0,250)
	#ax.set_box_aspect(aspect = 3)
	#fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.savefig("plots/output_heatmap_%s.png"%label)
	plt.close()
	
'''


gpu_boole = torch.cuda.is_available()
print("Is GPU available? ",gpu_boole)
print("Loading model... ")

data = pd.read_hdf("events_LHCO2020_BlackBox1.h5", stop = 5)
data = data.to_numpy()

test_set = torch.tensor(data, dtype=torch.float32)

test_loader = torch.utils.data.DataLoader(dataset = test_set,
	batch_size = 1,
	shuffle = False)

# LOAD AN EXISTING MODEL (POSSIBLE BUG)
model = AE()
if gpu_boole: model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),
        lr = 1e-3,
        weight_decay = 1e-8)

loss_function = torch.nn.MSELoss()

checkpoint = torch.load("checkpoints/ae_epoch4_%s.pth"%(ending))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loaded_epoch = checkpoint['epoch']
print("loaded epoch = ",loaded_epoch)
loaded_loss_function = checkpoint['loss']
print("loaded loss = ",loaded_loss_function)
'''
train_val_losses = []

with open("losses/train_val_losses_%s.txt"%ending,"r") as f:
	for line in f:
		train_val_losses.append(line.split(' '))
train_val_losses = np.array(train_val_losses).astype("float32")
#	print(train_val_losses)
losses = train_val_losses[:,0].tolist()
val_losses = train_val_losses[:,1].tolist()

plot_ae_loss(losses,val_losses)
'''
input_list, output_list = np.zeros((1,2100)), np.zeros((1,2100))
for i,input in enumerate(test_loader):
	if gpu_boole:
		input = input.cuda()

  
	output = model.forward(input)
	test_loss = loss_function(output, input)
	print("Loss for this input: ",test_loss.cpu().data.numpy().item())
	input_list = np.vstack((input_list,input.cpu().detach().numpy()))
	output_list = np.vstack((output_list,output.cpu().detach().numpy()))

np.savetxt("test_input_%s.txt"%(ending), input_list)
np.savetxt("test_output_%s.txt"%(ending), output_list)
'''

input_list = np.loadtxt("test_input_%s.txt"%(ending))[1:]
output_list = np.loadtxt("test_output_%s.txt"%(ending))[1:]
for i in range(len(input_list)):
	plot_heatmap(input_list[i], output_list[i], ending+'_%i'%i)
