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
ending = "092823_chamfer"
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
#print(data[0],data[1],data[2])

#print(data.shape)
#print(np.amin(data[:,0]),np.amin(data[:,3]),np.amin(data[:,1]),np.amin(data[:,4]),np.amin(data[:,2]),np.amin(data[:,5]))
#print(np.amax(data[:,0]),np.amax(data[:,3]),np.amax(data[:,1]),np.amax(data[:,4]),np.amax(data[:,2]),np.amax(data[:,5]))
#train, validate, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])

data = data.reshape((1000000,228,3))
data_unnorm = np.copy(data)
eta = data[:,:,0]
phi = data[:,:,1]
pt = data[:,:,2]
transformed = np.zeros((1000000,228,3))

print("Min max of eta, phi, pt before any transformations: ")
print(np.amin(eta),np.amax(eta))
print(np.amin(phi),np.amax(phi))
print(np.amin(pt),np.amax(pt))
px = np.multiply(pt,np.cos(phi))
py = np.multiply(pt,np.sin(phi))
pz = np.multiply(pt,np.sinh(eta))
E = np.multiply(pt,np.cosh(eta))
# shift range to 0->max
px_min, py_min, pz_min, E_min =  np.amin(px),np.amin(py),np.amin(pz),np.amin(E)
#px -= np.amin(px)
#py -= np.amin(py)
#pz -= np.amin(pz)
#E -= np.amin(E)
print("Min max of px, py, pz, E after shifting range, before scaling:")
print(np.amin(px), np.amax(px), np.mean(px))
print(np.amin(py), np.amax(py), np.mean(py))
print(np.amin(pz), np.amax(pz), np.mean(pz))
print(np.amin(E), np.amax(E), np.mean(E))
#standard scaling
px_mean, py_mean, pz_mean, E_mean = np.mean(px), np.mean(py), np.mean(pz), np.mean(E) #note that this is the mean after shifting the range
px_std, py_std, pz_std, E_std = np.std(px), np.std(py), np.std(pz), np.std(E)
px_scaled = (px - np.mean(px))/np.std(px)
py_scaled = (py - np.mean(py))/np.std(py)
pz_scaled = (pz - np.mean(pz))/np.std(pz)
E_scaled = (E - np.mean(E))/np.std(E)
'''
px_norm = px/np.amax(px)
py_norm = py/np.amax(py)
pz_norm = pz/np.amax(pz)
E_norm = E/np.amax(E)
print("Min max of px, py, pz after scaling:")
print(np.amin(px_scaled), np.amax(px_scaled), np.mean(px_scaled))
print(np.amin(py_scaled), np.amax(py_scaled), np.mean(py_scaled))
print(np.amin(pz_scaled), np.amax(pz_scaled), np.mean(pz_scaled))
print(np.amin(E_scaled), np.amax(E_scaled), np.mean(E_scaled))



print(np.amin(data[:,:,0]), np.amax(data[:,:,0]), np.mean(data[:,:,0])) 
print(np.amin(data[:,:,1]), np.amax(data[:,:,1]), np.mean(data[:,:,1]))
print(np.amin(data[:,:,2]), np.amax(data[:,:,2]), np.mean(data[:,:,2]))
data[:,:,0] = data[:,:,0]/np.amax(data[:,:,0])
data[:,:,1] = data[:,:,1]/np.amax(data[:,:,1])
data[:,:,2] = data[:,:,2]/np.amax(data[:,:,2])


pt_max = np.amax(data[:,:,0],axis=1)
eta_max = np.amax(data[:,:,1],axis=1)
phi_max = np.amax(data[:,:,2],axis=1)
data[:,:,0] = [np.zeros((159,)) if 0.0==pt_max[i] else data[i,:,0]/pt_max[i] for i in range(1000000)]
data[:,:,1] = [np.zeros((159,)) if 0.0==eta_max[i] else data[i,:,1]/eta_max[i] for i in range(1000000)]
data[:,:,2] = [np.zeros((159,)) if 0.0==phi_max[i] else data[i,:,2]/phi_max[i] for i in range(1000000)]
'''
del pt,eta,phi
gc.collect()
transformed[:,:,0] = px_scaled
transformed[:,:,1] = py_scaled
transformed[:,:,2] = pz_scaled
#transformed[:,:,3] = E_scaled
#print(pt_max[:30],eta_max[:30],phi_max[:30])
#data = data.reshape((1000000,684))
transformed = transformed.reshape((1000000,228*3))
train, validate, test = np.split(transformed, [int(.6*len(transformed)), int(.8*len(transformed))])

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
			#torch.nn.Linear(228*3 800),
			#torch.nn.ReLU(),
			#torch.nn.Linear(800, 700),
			#torch.nn.ReLU(),
			torch.nn.Linear(228*3, 600),
			torch.nn.ReLU(),
                        #torch.nn.Linear(600, 500),
                        #torch.nn.ReLU(),
                        torch.nn.Linear(600, 400),
                        torch.nn.ReLU(),
			#torch.nn.Linear(400, 300),
			#torch.nn.ReLU(),
			torch.nn.Linear(400, 200),
			torch.nn.ReLU(),
			#torch.nn.Linear(200, 100),
			#torch.nn.ReLU(),
			torch.nn.Linear(200, 100),
			torch.nn.ReLU(),
			torch.nn.Linear(100, 25),
			#torch.nn.ReLU(),
			#torch.nn.Linear(25, 10)
			)

		# DECODER
		self.decoder = torch.nn.Sequential(
			#torch.nn.Linear(10, 25),
			#torch.nn.ReLU(),
			torch.nn.Linear(25, 100),
			torch.nn.ReLU(),
			#torch.nn.Linear(50, 100),
			#torch.nn.ReLU(),
			torch.nn.Linear(100, 200),
			torch.nn.ReLU(),
			#torch.nn.Linear(200, 300),
			#torch.nn.ReLU(),
			torch.nn.Linear(200, 400),
			torch.nn.ReLU(),
			#torch.nn.Linear(400, 500),
                        #torch.nn.ReLU(),
                        torch.nn.Linear(400, 600),
                        torch.nn.ReLU(),
                       	#torch.nn.Linear(600, 700),
                        #torch.nn.ReLU(),
                        torch.nn.Linear(600, 228*3),
                        #torch.nn.ReLU(),
                        #torch.nn.Linear(800, 228*4),
                         
			)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

model = AE()
if gpu_boole: model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),
        lr = 2e-3,
        weight_decay = 1e-8)

#loss_function = torch.nn.MSELoss()
#loss_function = chamfer_distance()

# LOAD AN EXISTING MODEL 
if load_model:
	checkpoint = torch.load("checkpoints/ae_epoch2_%s.pth"%(ending))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	loaded_epoch = checkpoint['epoch']
	print("loaded epoch = ",loaded_epoch)
	#loss_function = checkpoint['loss']
	#print("loaded loss = ",loss_function)
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
				event = torch.reshape(event, (batch_size,228,3))
				reconstructed = torch.reshape(reconstructed, (batch_size,228,3))
				#loss = loss_function(reconstructed, event)
				loss,_ = chamfer_distance(reconstructed, event)
			 
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
	            	#'loss': loss_function
			},
			"checkpoints/ae_epoch%i_%s.pth"%(epoch%5,ending))
		losses.append(this_loss)
		print("Train Loss: %f"%(this_loss))
		
		# VALIDATION

		for event in val_loader:
			model.eval()
			if gpu_boole:
				event = event.cuda()
			
			
			reconstructed = model.forward(event)
			event = torch.reshape(event, (batch_size,228,3))
			reconstructed = torch.reshape(reconstructed, (batch_size,228,3))
			#val_loss = loss_function(reconstructed, event)
			
			val_loss,_ = chamfer_distance(reconstructed, event)
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
		if gpu_boole:
			event = event.cuda()

	  
		reconstructed = model.forward(event)
		event = torch.reshape(event, (1,228,3))
		reconstructed = torch.reshape(reconstructed, (1,228,3))
		#test_loss = loss_function(reconstructed, event)
		test_loss,_ = chamfer_distance(reconstructed,event)
		test_loss_per_epoch += test_loss.cpu().data.numpy().item()
		if idx < 2000:
			event = torch.reshape(event, (1,228*3))
			reconstructed = torch.reshape(reconstructed, (1,228*3))			
			#print("Loss for this input: ",test_loss.cpu().data.numpy().item())
			input_list = np.vstack((input_list,(event.cpu().detach().numpy())))
			output_list = np.vstack((output_list,(reconstructed.cpu().detach().numpy())))

	test_losses.append(test_loss_per_epoch/int(test.shape[0]))
	print("Test Loss: %f"%(test_loss_per_epoch/int(test.shape[0])))
	input_list = input_list[1:].reshape((2000,228,3))
	input_list[:,:,0] = (input_list[:,:,0] * px_std) + px_mean + px_min
	input_list[:,:,1] = (input_list[:,:,1] * py_std) + py_mean + py_min
	input_list[:,:,2] = (input_list[:,:,2] * pz_std) + pz_mean + pz_min
	#input_list[:,:,3] = (input_list[:,:,3] * E_std) + E_mean + E_min
	phi = np.arctan(np.divide(input_list[:,:,1],input_list[:,:,0]))
	pt = np.abs(np.divide(input_list[:,:,1],np.sin(phi)))
	eta = np.arcsinh(np.divide(input_list[:,:,2],pt))
	input_list_2 = np.zeros((2000,228,3))
	input_list_2[:,:,0],  input_list_2[:,:,1],  input_list_2[:,:,2] =  input_list[:,:,0], input_list[:,:,1],  input_list[:,:,2]
	#input_list[:,:,0], input_list[:,:,1], input_list[:,:,2] = pt, eta, phi
	input_list = input_list_2.reshape((2000,228*3))
	print("Min max of eta, phi, pt in input (test): ")
	print(np.amin(eta),np.amax(eta))
	print(np.amin(phi),np.amax(phi))
	print(np.amin(pt),np.amax(pt))	
	
	output_list = output_list[1:].reshape((2000,228,3))
	output_list[:,:,0] = (output_list[:,:,0] * px_std) + px_mean + px_min
	output_list[:,:,1] = (output_list[:,:,1] * py_std) + py_mean + py_min
	output_list[:,:,2] = (output_list[:,:,2] * pz_std) + pz_mean + pz_min	
	phi = np.arctan(np.divide(output_list[:,:,1],output_list[:,:,0]))
	pt = np.abs(np.divide(output_list[:,:,1],np.sin(phi)))
	eta = np.arcsinh(np.divide(output_list[:,:,2],pt))
	output_list_2 = np.zeros((2000,228,3))
	output_list_2[:,:,0], output_list_2[:,:,1], output_list_2[:,:,2] = output_list[:,:,0], output_list[:,:,1],  output_list[:,:,2]
	output_list = output_list_2.reshape((2000,684))
	print("Min max of eta, phi, pt in output (test): ")
	print(np.amin(eta),np.amax(eta))
	print(np.amin(phi),np.amax(phi))	
	print(np.amin(pt),np.amax(pt))
	'''
	data = data_unnorm.reshape((1000000,228,3))
	print(input_list.shape)
	input_list = input_list[1:]
	print(input_list.shape)
	input_list = input_list.reshape((20,228,3))
	print(input_list.shape,)
	input_list[:,:,0] = input_list[:,:,0]*np.amax(data[:,:,0])
	input_list[:,:,1] = input_list[:,:,1]*np.amax(data[:,:,1])
	input_list[:,:,2] = input_list[:,:,2]*np.amax(data[:,:,2])
	input_list = input_list.reshape((20,684))
        
	output_list = output_list[1:].reshape((20,228,3))
	output_list[:,:,0] = output_list[:,:,0]*np.amax(data[:,:,0])
	output_list[:,:,1] = output_list[:,:,1]*np.amax(data[:,:,1])
	output_list[:,:,2] = output_list[:,:,2]*np.amax(data[:,:,2])
	output_list = output_list.reshape((20,684))
	'''
	
	np.savetxt("test_input_%s.txt"%(ending), input_list)
	np.savetxt("test_output_%s.txt"%(ending), output_list)


