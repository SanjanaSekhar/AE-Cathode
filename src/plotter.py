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
import pandas as pd


ending = "031723"



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

def plot_3D(input, output, label):
	
	input = input.reshape(159,3)
	output = output.reshape(159,3)
		
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
	ax.set_ylim(0,1)
	ax.set_xlim(0,1)
	#ax.set_xlim(-3.2,3.2)
	#ax.set_ylim(0,6.5)
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
	

train_val_losses = []

with open("losses/train_val_losses_%s.txt"%ending,"r") as f:
	for line in f:
		train_val_losses.append(line.split(' '))
train_val_losses = np.array(train_val_losses).astype("float32")
#	print(train_val_losses)
losses = train_val_losses[:,0].tolist()
val_losses = train_val_losses[:,1].tolist()

plot_ae_loss(losses,val_losses)

input_list = np.loadtxt("test_input_%s.txt"%(ending))[:,:477]
output_list = np.loadtxt("test_output_%s.txt"%(ending))[:,:477]
for i in range(len(input_list)):
	plot_3D(input_list[i], output_list[i], ending+'_%i'%i)
