import time
from time import sleep
from tqdm import tqdm
import struct
import math
## External Library
import numpy as np
import matplotlib.pyplot as plt

## Pytorch Imports
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

ending = "031723"
load_model = True
test_model = True
early_stop = 5
batch_size = 512
epochs = 200

gpu_boole = torch.cuda.is_available()
print("Is GPU available? ",gpu_boole)
if load_model: print("Loading model... ")

data = pd.read_hdf("events_LHCO2020_BlackBox1_preprocessed.h5")
data = data.to_numpy()[:,:477]
#print(data.shape)
#print(np.amin(data[:,0]),np.amin(data[:,3]),np.amin(data[:,1]),np.amin(data[:,4]),np.amin(data[:,2]),np.amin(data[:,5]))
#print(np.amax(data[:,0]),np.amax(data[:,3]),np.amax(data[:,1]),np.amax(data[:,4]),np.amax(data[:,2]),np.amax(data[:,5]))
#train, validate, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])

data = data.reshape((1000000,159,3))
print(np.amin(data[:,:,0]), np.amax(data[:,:,0]), np.mean(data[:,:,0])) 
print(np.amin(data[:,:,1]), np.amax(data[:,:,1]), np.mean(data[:,:,1]))
print(np.amin(data[:,:,2]), np.amax(data[:,:,2]), np.mean(data[:,:,2]))
data[:,:,0] = data[:,:,0]/np.amax(data[:,:,0])
data[:,:,1] = data[:,:,1]/np.amax(data[:,:,1])
data[:,:,2] = data[:,:,2]/np.amax(data[:,:,2])
data = data.reshape((1000000,477))

train, validate, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])

train_set = torch.tensor(train, dtype=torch.float32)
val_set = torch.tensor(validate, dtype=torch.float32)
test_set = torch.tensor(test, dtype=torch.float32)

train_loader = torch.utils.data.DataLoader(dataset = train_set,
	batch_size = batch_size,
	shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_set,
	batch_size = batch_size,
	shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_set,
	batch_size = 1,
	shuffle = True)

# AUTOENCODER CLASS

class AE(torch.nn.Module):
	def __init__(self):
		super().__init__()

		# ENCODER
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(477, 400),
			torch.nn.ReLU(),
			torch.nn.Linear(400, 300),
			torch.nn.ReLU(),
			torch.nn.Linear(300, 200),
			torch.nn.ReLU(),
			torch.nn.Linear(200, 100),
			torch.nn.ReLU(),
			torch.nn.Linear(100, 50),
			torch.nn.ReLU(),
			torch.nn.Linear(50, 25),
			torch.nn.ReLU(),
			torch.nn.Linear(25, 10)
			)

		# DECODER
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(10, 25),
			torch.nn.ReLU(),
			torch.nn.Linear(25, 50),
			torch.nn.ReLU(),
			torch.nn.Linear(50, 100),
			torch.nn.ReLU(),
			torch.nn.Linear(100, 200),
			torch.nn.ReLU(),
			torch.nn.Linear(200, 300),
			torch.nn.ReLU(),
			torch.nn.Linear(300, 400),
			torch.nn.ReLU(),
			torch.nn.Linear(400, 477)
		
			)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

model = AE()
if gpu_boole: model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),
        lr = 1e-3,
        weight_decay = 1e-8)

loss_function = torch.nn.MSELoss()


# LOAD AN EXISTING MODEL (POSSIBLE BUG)
if load_model:
	checkpoint = torch.load("checkpoints/ae_epoch4_%s.pth"%(ending))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	loaded_epoch = checkpoint['epoch']
	print("loaded epoch = ",loaded_epoch)
	loss_function = checkpoint['loss']
	print("loaded loss = ",loss_function)
	train_val_losses = []
	
	with open("losses/train_val_losses_%s.txt"%ending,"r") as f:
		for line in f:
			train_val_losses.append(line.split(' '))
	train_val_losses = np.array(train_val_losses).astype("float32")
#	print(train_val_losses)
	losses = train_val_losses[:,0].tolist()
	val_losses = train_val_losses[:,1].tolist()
	'''
	losses,val_losses = [],[]
	'''
	
else:
	loaded_epoch = 0
	losses,val_losses = [],[]
	


print("train shape ",train.shape)
print("val shape ",validate.shape)
print("test shape ",test.shape)


outputs = []
test_losses = []

# TRAINING & VALIDATION LOOP

if not test_model:
	
	for epoch in range(loaded_epoch,epochs):

		loss_per_epoch, val_loss_per_epoch = 0,0
		i = 0
		with tqdm(train_loader, unit="batch") as tepoch:
			model.train()
			for event in tepoch:
				tepoch.set_description(f"Epoch {epoch}")
				if gpu_boole:
					event = event.cuda()

			  	# Output of Autoencoder
				reconstructed = model.forward(event)

			  	# Calculating the loss function
				loss = loss_function(reconstructed, event)

			 
				#if epoch > 0 and epoch != loaded_epoch:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			 	 # Adding up all losses in a batch
				#i+=1
				#print("loss %i = "%i,loss)
				#print("loss.cpu().data.numpy().item() = ",loss.cpu().data.numpy().item())
				loss_per_epoch += loss.cpu().data.numpy().item()
				sleep(0.1)
		
		this_loss = loss_per_epoch/math.ceil(train.shape[0]/batch_size)
		torch.save({
			'epoch':epoch,
			'model_state_dict': model.state_dict(),
	            	'optimizer_state_dict': optimizer.state_dict(),
	            	'loss': loss_function},
			"checkpoints/ae_epoch%i_%s.pth"%(epoch%5,ending))
		losses.append(this_loss)
		print("Train Loss: %f"%(this_loss))
		
		# VALIDATION

		for event in val_loader:
			model.eval()
			if gpu_boole:
				event = event.cuda()

			reconstructed = model.forward(event)
			val_loss = loss_function(reconstructed, event)
			val_loss_per_epoch += val_loss.cpu().data.numpy().item()

		val_losses.append(val_loss_per_epoch/math.ceil(validate.shape[0]/batch_size))
		print("Val Loss: %f"%(val_loss_per_epoch/math.ceil(validate.shape[0]/batch_size)))
		
		# EARLY STOPPING
		flag = 0
		if early_stop > 0 and epoch > loaded_epoch + early_stop:
			for e in range(early_stop):
				if val_losses[-e] > val_losses[-e-1]: flag += 1
			if flag == early_stop:
				print("STOPPING TRAINING EARLY, VAL LOSS HAS BEEN INCREASING FOR THE LAST %i EPOCHS"%early_stop)
				break

		with open("losses/train_val_losses_%s.txt"%ending,"w") as f:
			for loss, val_loss in zip(losses, val_losses):
				f.write(str(loss)+" "+str(val_loss)+"\n")

	print("========== TRAINING COMPLETE ===========")

# TESTING
if test_model:

	test_loss_per_epoch = 0.
	input_list, output_list = np.zeros((1,477)), np.zeros((1,477))
	for idx,event in enumerate(test_loader):
		if gpu_boole:
			event = event.cuda()

	  
		reconstructed = model.forward(event)
		test_loss = loss_function(reconstructed, event)
		test_loss_per_epoch += test_loss.cpu().data.numpy().item()
		if idx < 20:
			print("Loss for this input: ",test_loss.cpu().data.numpy().item())
			input_list = np.vstack((input_list,(event.cpu().detach().numpy())))
			output_list = np.vstack((output_list,(reconstructed.cpu().detach().numpy())))

	test_losses.append(test_loss_per_epoch/int(test.shape[0]))
	print("Test Loss: %f"%(test_loss_per_epoch/int(test.shape[0])))

	np.savetxt("test_input_%s.txt"%(ending), input_list[1:])
	np.savetxt("test_output_%s.txt"%(ending), output_list[1:])


