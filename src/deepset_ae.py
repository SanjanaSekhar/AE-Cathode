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
from fspool import FSPool

ending = "012324"
load_model = True
test_model = True
early_stop = 5
batch_size = 628
epochs = 20

gpu_boole = torch.cuda.is_available()
print("Is GPU available? ",gpu_boole)
if load_model: print("Loading model... ")

#data = pd.read_csv("events_rotated.csv", header=None,usecols=np.arange(0,684))
#print(data.head())
data = pd.read_hdf("events_LHCO2020_BlackBox1_preprocessed_rotated.h5",stop=1000000)
data = data.to_numpy()[:,:684]


data = data.reshape((1000000,228,3))
data_unnorm = np.copy(data)
eta = np.copy(data[:,:,0])
phi = np.copy(data[:,:,1])
pt = np.copy(data[:,:,2])
print(eta.shape, eta[0])
print(phi.shape, phi[0])
print(pt.shape, pt[0])
# Add PID : 0 for zero pt particles, 1 otherwise
pid = np.asarray([(np.array(p) != 0).astype(int) for p in pt])
#pid = np.vstack(pid, np.logical_not(pid).astype(int))
print("PID shape: ",pid.shape) 


# take log10 of pt
logpt = np.zeros((1000000,228))
logpt[pt!=0] = np.log10(pt[pt!=0])


data[:,:,1] = np.copy(eta/np.max(eta))
data[:,:,2] = np.copy(phi/np.max(phi))
data[:,:,0] = np.copy(pt/np.max(pt))
'''
data[:,:,1] = np.copy(eta)
data[:,:,2] = np.copy(phi)
data[:,:,0] = np.copy(pt)
'''
data = np.dstack((data, pid))
print("Data shape after stacking PID", data.shape)
pid = np.vstack((pid, np.logical_not(pid).astype(int)))
print("\n",data[0,0,0],"\n",data[0,1,0],"\n",data[0,2,0],"\n",data[0,3,0])
data = np.swapaxes(data, 1, 2)
print("Data shape after swapping axes", data.shape)
train, validate, test = np.split(data, [int(.7*len(data)), int(.8*len(data))])


print("\n",data[0,0,0],"\n",data[0,1,0],"\n",data[0,2,0],"\n",data[0,3,0])


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
			torch.nn.Conv1d(4,50,1),
			torch.nn.ELU(),
			torch.nn.Conv1d(50,100,1),
			torch.nn.ELU(),
			torch.nn.Conv1d(100,50,1),
			torch.nn.ELU(),
			torch.nn.Conv1d(50, 11-1, 1)) # 9 latent dimensions + 1 mask

		self.pool = FSPool(11 -1, 30, relaxed=False) # second argument is no. of points needed to parametrize a piecewise linear function, can be arbit
		

		# DECODER
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(11, 100),
			torch.nn.ELU(),
			torch.nn.Linear(100, 400),
			torch.nn.ELU(),
			torch.nn.Linear(400, 700),
			torch.nn.ELU(),
                        torch.nn.Linear(700, 1000),
                        torch.nn.ELU(),
			#torch.nn.Linear(400, 500),
			#torch.nn.ReLU(),
			#torch.nn.Linear(500, 600),
			#torch.nn.ReLU(),
                        )
		self.regress_4vec = torch.nn.Linear(1000, 228*3) # 4 vectors of 228 particles
		self.classif = torch.nn.Linear(1000, 228*1) # mask choice of 0 or 1 for 228 particles
		self.mask_out = torch.nn.Sigmoid()        

	def forward(self, x, mask):
		encoded = self.create_deepset(x)
		#print("encoder o/p shape = ", encoded.shape, "mask.unsqueeze(1).shape = ",mask.unsqueeze(1).shape)
		encoded = encoded * mask.unsqueeze(1) # mask away invalid elements
		#print("encoder o/p shape after mask = ", encoded.shape)
		pooled, _ = self.pool(encoded)
		#print("pooled o/p shape = ", pooled.shape)
		#print("pooled.mean = ",pooled.mean()," mask.mean(dim=1) = ",mask.mean(dim=1))
		pooled = torch.cat([pooled,mask.unsqueeze(1).mean(dim=2)],dim=1) # save mask to the representation
		#print("pooled shape after cat mask = ", pooled.shape)
		decoded = self.decoder(pooled)
		vec = self.regress_4vec(decoded)
		vec = vec.view(vec.size(0), 3, 228) #return 4 vectors in (n_events, 4, 228) shape
		mask = self.mask_out(self.classif(decoded))
		mask = mask.view(mask.size(0), 228)
		return vec, mask 

model = DeepSetAE()
if gpu_boole: model = model.cuda()

optimizer = torch.optim.Adamax(model.parameters(),
        lr = 1e-2,
        weight_decay = 1e-6)

BCE_loss = torch.nn.BCELoss()
alpha = 0.75
epoch = 16
alpha_list = [0.75]
#alpha_list = np.linspace(0.0,0.1,10)
#loss_function = chamfer_distance()

# LOAD AN EXISTING MODEL 
if load_model:
	checkpoint = torch.load("checkpoints/deepset_alpha%.2f_epoch%i_%s.pth"%(alpha,epoch,ending))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	loaded_epoch = checkpoint['epoch']
	print("loaded epoch = ",loaded_epoch)
	#loss_function = checkpoint['loss']
	#print("loaded loss = ",loss_function)
	train_val_losses = []
	scale_c = checkpoint['scale_c']
	scale_b = checkpoint['scale_b']

	with open("losses/deepset_alpha%.2f_train_val_losses_%s.txt"%(alpha,ending),"r") as f:
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
	for alpha in alpha_list:
		if not load_model: losses,val_losses = [],[]	
		best_train_loss, best_val_loss = 1e7,1e7
		for epoch in range(loaded_epoch,epochs):

			closs_per_epoch, bloss_per_epoch, loss_per_epoch, val_loss_per_epoch = 0,0,0,0
			i = 0
			with tqdm(train_loader, unit="batch") as tepoch:
				model.train()
				for event in tepoch:
					tepoch.set_description(f"Epoch {epoch}")
					if gpu_boole:
						event = event.cuda()
					#print(event.shape)
					#event = torch.reshape(event, (batch_size,228,4))
					event_vec = event[:,0:3,:]
					event_mask = event[:,3,:]
				  	# Output of Autoencoder
					reconstructed_vec, reconstructed_mask = model.forward(event,event_mask)

				  	# Calculating the loss function
					
					#reconstructed_vec = torch.reshape(reconstructed_vec, (batch_size,228,3))
					#loss = loss_function(reconstructed, event)
					chamfer_loss,_ = chamfer_distance(reconstructed_vec, event_vec)
					bce_loss = BCE_loss(reconstructed_mask, event_mask)
					if epoch == 0: scale_c, scale_b = chamfer_loss.cpu().data.numpy().item(), bce_loss.cpu().data.numpy().item()
					loss = alpha * (chamfer_loss/scale_c) + (1-alpha) * (bce_loss/scale_b)
					#if epoch > 0 and epoch != loaded_epoch:
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				 	 # Adding up all losses in a batch
					#i+=1
					#print("loss %i = "%i,loss)
					#print("loss.cpu().data.numpy().item() = ",loss.cpu().data.numpy().item())
					closs_per_epoch += (chamfer_loss.cpu().data.numpy().item())/batch_size
					bloss_per_epoch += (bce_loss.cpu().data.numpy().item())/batch_size
					loss_per_epoch += loss.cpu().data.numpy().item()
					sleep(0.1)
			
			this_loss = loss_per_epoch/math.ceil(train.shape[0]/batch_size)
			losses.append(this_loss)
			print("Chamfer Loss = %f, BCE Loss = %f, Train Loss: %f"%(closs_per_epoch,bloss_per_epoch,this_loss))
			
			# VALIDATION

			for event in val_loader:
				model.eval()
				if gpu_boole:
					event = event.cuda()
				
				# Output of Autoencoder
				#event = torch.reshape(event, (batch_size,228,4))
				event_vec = event[:,0:3,:]
				event_mask = event[:,3,:]
				reconstructed_vec, reconstructed_mask = model.forward(event, event_mask)
				
				#reconstructed_vec = torch.reshape(reconstructed_vec, (batch_size,228,3))
				#loss = loss_function(reconstructed, event)
				chamfer_loss,_ = chamfer_distance(reconstructed_vec, event_vec)
				bce_loss = BCE_loss(reconstructed_mask, event_mask)
				val_loss = alpha * (chamfer_loss/scale_c) + (1-alpha) * (bce_loss/scale_b)
				
				val_loss_per_epoch += val_loss.cpu().data.numpy().item()

			val_losses.append(val_loss_per_epoch/math.ceil(validate.shape[0]/batch_size))
			print("Val Loss: %f"%(val_loss_per_epoch/math.ceil(validate.shape[0]/batch_size)))
			
			# SAVE BEST MODEL
			if this_loss < best_train_loss and val_losses[-1] < best_val_loss:
				best_train_loss, best_val_loss = this_loss, val_losses[-1]
				torch.save({
					'epoch':epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': loss,
					'scale_c': scale_c,
					'scale_b': scale_b
					},
					"checkpoints/deepset_alpha%.2f_epoch%i_%s.pth"%(alpha,epoch,ending))
			# EARLY STOPPING
			flag = 0
			if early_stop > 0 and epoch > loaded_epoch + early_stop:
				for e in range(1,early_stop+1):
					if val_losses[-e] > val_losses[-early_stop]: flag += 1
				if flag == early_stop:
					print("STOPPING TRAINING EARLY, VAL LOSS HAS BEEN INCREASING FOR THE LAST %i EPOCHS"%early_stop)
					break

			with open("losses/deepset_alpha%.2f_train_val_losses_%s.txt"%(alpha,ending),"w") as f:
				for loss, val_loss in zip(losses, val_losses):
					f.write(str(loss)+" "+str(val_loss)+"\n")

		print("========== TRAINING COMPLETE for alpha = %.2f ==========="%alpha)

# TESTING
if test_model:

	test_loss_per_epoch = 0.
	input_list, output_list = np.zeros((1,4,228)), np.zeros((1,4,228))
	for idx,event in enumerate(test_loader):
		#if idx==0: print(event.numpy())
		if gpu_boole:
			event = event.cuda()

	  	# Output of Autoencoder
		#event = torch.reshape(event, (batch_size,228,4))
		event_vec = event[:,0:3,:]
		event_mask  = event[:,3,:]
		reconstructed_vec, reconstructed_mask = model.forward(event, event_mask)
		#reconstructed_vec = torch.reshape(reconstructed_vec, (1,228,3))
		#loss = loss_function(reconstructed, event)
		chamfer_loss,_ = chamfer_distance(reconstructed_vec, event_vec)
		bce_loss = BCE_loss(reconstructed_mask, event_mask)
		test_loss = alpha * (chamfer_loss/scale_c) + (1-alpha) * (bce_loss/scale_b)
		test_loss_per_epoch += test_loss.cpu().data.numpy().item()
		if idx < 2000:
			reconstructed_mask = reconstructed_mask.unsqueeze(1)
			reconstructed = torch.cat([reconstructed_vec,reconstructed_mask],dim=1)
			#print(reconstructed_vec.shape, reconstructed_mask.shape, reconstructed.shape)
			#event_vec = torch.reshape(event_vec, (1,228*3))
			#reconstructed_vec = torch.reshape(reconstructed_vec, (228,3))	
			#reconstructed_mask = torch.reshape(reconstructed_mask,(228,1))
			#reconstructed_vec = reconstructed_vec * torch.sigmoid(reconstructed_mask.unsqueeze(1))
			#reconstructed_vec = torch.reshape(reconstructed_vec, (1,3,228))		
			#print("Loss for this input: ",test_loss.cpu().data.numpy().item())
			input_list = np.vstack((input_list,(event.cpu().detach().numpy())))
			output_list = np.vstack((output_list,(reconstructed.cpu().detach().numpy())))
			#if idx==0: print(event.cpu().detach().numpy())
	
	test_losses.append(test_loss_per_epoch/int(test.shape[0]))
	print("Test Loss: %f"%(test_loss_per_epoch/int(test.shape[0])))
	print(input_list.shape)
	print((input_list[:,0,:]!=0).shape)
	#pt = input_list[:,0,:] 
	#pt[pt!=0] = 10**pt[pt!=0]
	#input_list[input_list[:,0,:]!=0] = 10**input_list[input_list[:,0,:]!=0]
	input_list[:,0,:] = np.copy(input_list[:,0,:]*np.amax(pt))
	input_list[:,3,:] = np.copy(np.round(input_list[:,3,:]))
	input_list[:,0,:], input_list[:,1,:], input_list[:,2,:], input_list[:,3,:] = np.copy(np.multiply(input_list[:,0,:],input_list[:,3,:])), np.copy(np.multiply(input_list[:,1,:],input_list[:,3,:])), np.copy(np.multiply(input_list[:,2,:],input_list[:,3,:])), np.copy(input_list[:,3,:])	
	input_list = input_list.reshape((2001,228*4))
	np.savetxt("deepset_test_input_ptetaphi_%s.txt"%(ending), input_list[1:])
	#output_list = np.swapaxes(output_list,1,2)
	#pt = output_list[:,0,:]
	#pt[pt!=0] = 10**pt[pt!=0]
	#output_list[:,output_list[:,0,:]!=0,:] = 10**output_list[:,output_list[:,0,:]!=0,:]
	output_list[:,0,:] = np.copy(output_list[:,0,:]*np.amax(pt))
	output_list[:,3,:] = np.copy(np.round(output_list[:,3,:]))
	output_list[:,0,:], output_list[:,1,:], output_list[:,2,:], output_list[:,3,:] = np.copy(np.multiply(output_list[:,0,:],output_list[:,3,:])),  np.copy(np.multiply(output_list[:,1,:],output_list[:,3,:])),  np.copy(np.multiply(output_list[:,2,:],output_list[:,3,:])), np.copy(output_list[:,3,:])
	
	output_list = output_list.reshape((2001,228*4))
	np.savetxt("deepset_test_output_ptetaphi_%s.txt"%(ending), output_list[1:])
	print(output_list[1, :228])
	print(output_list[1, 228:228*2])
	print(output_list[1, 228*2:228*3])	
	print(output_list[1, 228*3:])
