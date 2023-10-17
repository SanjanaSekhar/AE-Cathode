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
torch.cuda.empty_cache()
import gc
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch3d.loss import chamfer_distance

ending = "100423"
load_model = True
test_model = True
early_stop = 5
batch_size = 400
epochs = 300

gpu_boole = torch.cuda.is_available()
print("Is GPU available? ",gpu_boole)
if load_model: print("Loading model... ")

#data = pd.read_csv("events_rotated.csv", header=None,usecols=np.arange(0,684))
#print(data.head())
data = pd.read_hdf("events_LHCO2020_BlackBox1_preprocessed_rotated.h5",stop=1000000)
data = data.to_numpy()[:,:684]


data = data.reshape((1000000,228,3))
data_unnorm = np.copy(data)
eta = data[:,:,0]
phi = data[:,:,1]
pt = data[:,:,2]

# Add PID : 0 for zero pt particles, 1 otherwise
pid = np.asarray([(np.array(p) != 0).astype(int) for p in pt])
#pid = np.vstack(pid, np.logical_not(pid).astype(int))
print("PID shape: ",pid.shape) 


# take log10 of pt
logpt = np.zeros((1000000,228))
logpt[pt!=0] = np.log10(pt[pt!=0])


data[:,:,0] = eta/np.max(eta)
data[:,:,1] = phi/np.max(phi)
data[:,:,2] = logpt
data = np.dstack(data, pid)
print("Data shape after stacking PID", data.shape)
pid = np.vstack(pid, np.logical_not(pid).astype(int))

train, validate, test = np.split(data, [int(.7*len(data)), int(.8*len(data))])

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

class DeepSetAE(torch.nn.Module):
	def __init__(self):
		super().__init__()

		for m in self.modules():
	            if (
	                isinstance(m, torch.nn.Linear)
	                or isinstance(m, torch.nn.Conv2d)
	                or isinstance(m, torch.nn.Conv1d)
	            ):
	                torch.nn.init.xavier_uniform_(m.weight)
	                if m.bias is not None:
	                    m.bias.data.zero_()

		# ENCODER
		'''
		Create deep sets for each particle with inputs eta, phi, pt, mask
		Mask away inputs and pool using FSPool
		'''
		self.create_deepset = torch.nn.Sequential(
			torch.nn.Conv1d(4,128,1),
			torch.nn.ELU(),
			torch.nn.Conv1d(128,256,1),
			torch.nn.ELU(),
			torch.nn.Conv1d(256,128,1),
			torch.nn.ELU(),
			torch.nn.Conv1d(128, 11-1, 1)) # 9 latent dimensions + 1 mask

		self.pool = FSPool(11 -1, 20, relaxed=False) # second argument is no. of points needed to parametrize a piecewise linear function, can be arbit
		

		# DECODER
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(11, 100),
			torch.nn.ELU(),
			torch.nn.Linear(100, 200),
			torch.nn.ELU(),
			torch.nn.Linear(200, 400),
			torch.nn.ELU(),
                        torch.nn.Linear(400, 600),
                        torch.nn.ELU(),
                        )
		self.regress_4vec = torch.nn.Linear(600, 228*3) # 4 vectors of 228 particles
		self.classif_mask = torch.nn.Linear(600, 228*1) # mask choice of 0 or 1 for 228 particles
                        

	def forward(self, x, mask):
		encoded = self.create_deepset(x)
		print("encoder o/p shape = ", encoded.shape)
		encoded = torch.from_numpy(encoded.to_numpy() * mask[:,0]) # mask away invalid elements
		pooled, _ = self.pool(encoded)
		print("pooled o/p shape = ", pooled.shape)
		pooled = torch.cat([pooled,mask],dim=1) # save mask to the representation
		decoded = self.decoder(pooled)
		vec = self.regress_4vec(decoded)
		vec = vec.view(vec.size(0), 3, 228) #return 4 vectors in (n_events, 4, 228) shape
		mask = self.classif_mask(decoded)
		mask = mask.view(mask.size(0), 1, 228)
		return vec, mask 

model = DeepSetAE()
if gpu_boole: model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),
        lr = 2e-3,
        weight_decay = 1e-8)

BCE_loss = torch.nn.BCELoss()
alpha = 0.5
#loss_function = chamfer_distance()

# LOAD AN EXISTING MODEL 
if load_model:
	checkpoint = torch.load("checkpoints/deepsetAE_epoch2_%s.pth"%(ending))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	loaded_epoch = checkpoint['epoch']
	print("loaded epoch = ",loaded_epoch)
	#loss_function = checkpoint['loss']
	#print("loaded loss = ",loss_function)
	train_val_losses = []
	
	with open("losses/deepsetAE_train_val_losses_%s.txt"%ending,"r") as f:
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

				event = torch.reshape(event, (batch_size,228,4))
				event_vec = event[:,:,0:3]
				event_mask = event[:,:,3]
			  	# Output of Autoencoder
				reconstructed_vec, reconstructed_mask = model.forward(event,event_mask)

			  	# Calculating the loss function
				
				reconstructed_vec = torch.reshape(reconstructed_vec, (batch_size,228,3))
				#loss = loss_function(reconstructed, event)
				chamfer_loss,_ = chamfer_distance(reconstructed_vec, event_vec)
				bce_loss = BCE_loss(torch.sigmoid(reconstructed_mask), event_mask)
			 	loss = alpha * chamfer_loss + (1-alpha) * bce_loss
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
	            	'loss': loss
			},
			"checkpoints/deepsetAE_epoch%i_%s.pth"%(epoch%5,ending))
		losses.append(this_loss)
		print("Train Loss: %f"%(this_loss))
		
		# VALIDATION

		for event in val_loader:
			model.eval()
			if gpu_boole:
				event = event.cuda()
			
			# Output of Autoencoder
			event = torch.reshape(event, (batch_size,228,4))
			event_vec = event[:,:,0:3]
			event_mask = event[:,:,3]
			reconstructed_vec, reconstructed_mask = model.forward(event, event_mask)
			
			reconstructed_vec = torch.reshape(reconstructed_vec, (batch_size,228,3))
			#loss = loss_function(reconstructed, event)
			chamfer_loss,_ = chamfer_distance(reconstructed_vec, event_vec)
			bce_loss = BCE_loss(torch.sigmoid(reconstructed_mask), event_mask)
		 	val_loss = alpha * chamfer_loss + (1-alpha) * bce_loss
			
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
	input_list, output_list = np.zeros((1,228*3)), np.zeros((1,228*3))
	for idx,event in enumerate(test_loader):
		if idx==0: print(event.numpy())
		if gpu_boole:
			event = event.cuda()

	  	# Output of Autoencoder
	  	event = torch.reshape(event, (batch_size,228,4))
		event_vec = event[:,:,0:3]
		event_mask  = event[:,:,3]
		reconstructed_vec, reconstructed_mask = model.forward(event, event_mask)
		
		reconstructed_vec = torch.reshape(reconstructed_vec, (1,228,3))
		#loss = loss_function(reconstructed, event)
		chamfer_loss,_ = chamfer_distance(reconstructed_vec, event_vec)
		bce_loss = BCE_loss(torch.sigmoid(reconstructed_mask), event_mask)
	 	test_loss = alpha * chamfer_loss + (1-alpha) * bce_loss
		test_loss_per_epoch += test_loss.cpu().data.numpy().item()
		if idx < 2000:
			event_vec = torch.reshape(event_vec, (1,228*3))
			reconstructed_vec = torch.reshape(reconstructed_vec, (228,3))	
			reconstructed_mask = torch.reshape(reconstructed_mask,(228,1))
			reconstructed_vec = reconstructed_vec * reconstructed_mask
			reconstructed_vec = torch.reshape(reconstructed_vec, (1,228*3))		
			#print("Loss for this input: ",test_loss.cpu().data.numpy().item())
			input_list = np.vstack((input_list,(event.cpu().detach().numpy())))
			output_list = np.vstack((output_list,(reconstructed.cpu().detach().numpy())))
			if idx==0: print(event.cpu().detach().numpy())
	
	test_losses.append(test_loss_per_epoch/int(test.shape[0]))
	print("Test Loss: %f"%(test_loss_per_epoch/int(test.shape[0])))
	
	np.savetxt("deepset_test_input_ptetaphi_%s.txt"%(ending), input_list[1:])

	np.savetxt("deepset_test_output_ptetaphi_%s.txt"%(ending), output_list[1:])
	

