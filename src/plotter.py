import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from scipy.stats import norm
import numpy as np
#from matplotlib.backends.backend_pdf import PdfPages
#:from scipy import optimize
#import seaborn as sns
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


ending = "092823_chamfer"



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

def plot_3D(input, output, label, type = 'px'):
	
        input = input.reshape(228,3)
        output = output.reshape(228,3)
        if type == 'px':
            input = pd.DataFrame(input, columns = ["px","py","pz"])
            output = pd.DataFrame(output, columns = ["px","py","pz"])
        else:
            input = pd.DataFrame(input, columns = ["pT","eta","phi"])
            input = input.loc[~(input==0).all(axis=1)]
            output = pd.DataFrame(output, columns = ["pT","eta","phi"])
            output = output.loc[~(output==0).all(axis=1)]#print("============ INPUT ==============")
            print(np.amin(input['eta']),np.amax(input['eta']))
            print(np.amin(input['phi']),np.amax(input['phi']))
            print(np.amin(input['pT']),np.amax(input['pT']))
            #print("============ OUTPUT ==============")
	    #print(output)
        fig = plt.figure(figsize=(10,10))
        ax = fig.subplots(2,2,subplot_kw=dict(projection='3d'))
        #ax[0,0] = fig.gca(projection='3d')
        #ax[0,1] = fig.gca(projection='3d')
        #ax[1,0] = fig.gca(projection='3d')
        #ax[1,1] = fig.gca(projection='3d')
        #ax.set_box_aspect(aspect = 3)
        #surf = ax.plot_trisurf(input['eta'],input['phi'],input['pT'], cmap=plt.cm.coolwarm, linewidth=0.4)
        if type == 'px': surf0 = ax[0,0].bar3d(input['px'],input['py'],input['pz'].abs(),50,50,50,color='y',shade=True)
        else: surf0 = ax[0,0].bar3d(input['phi'],input['eta'],input['pT'].abs(),0.1,0.1,20,color='y',shade=True) 
        in_proxy = plt.Rectangle((0, 0), 1, 1, fc="y")
        if type == 'px': surf1 = ax[0,1].bar3d(output['px'],output['py'],output['pz'].abs(),10,10,20,color='b',shade=True)
        else: surf1 = ax[0,1].bar3d(output['phi'],output['eta'],output['pT'].abs(),0.1,0.1,20,color='b',shade=True)
        out_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
        if type == 'px': 
            surf2 = ax[1,0].bar3d(input['px'],input['py'],input['pz'].abs(),50,50,50,color='y',shade=True)
            surf3 = ax[1,0].bar3d(output['px'],output['py'],output['pz'].abs(),50,50,50,color='b',shade=True)
            surf4 = ax[1,1].bar3d(input['py'],input['px'],input['pz'].abs(),50,50,50,color='y',shade=True)
            surf5 = ax[1,1].bar3d(output['py'],output['px'],output['pz'].abs(),50,50,50,color='b',shade=True)
            for i in [0,1]:
                    for j in [0,1]:
                        ax[i,j].set_xlabel("px")
                        ax[i,j].set_ylabel("py")
                        ax[i,j].set_zlabel("pz")
            ax[1,1].set_xlabel("py")
            ax[1,1].set_ylabel("px")
            ax[1,1].set_zlabel("pz")
        else:
            surf2 = ax[1,0].bar3d(input['phi'],input['eta'],input['pT'].abs(),0.1,0.1,20,color='y',shade=True)
            surf3 = ax[1,0].bar3d(output['phi'],output['eta'],output['pT'].abs(),0.1,0.1,20,color='b',shade=True)
            surf4 = ax[1,1].bar3d(input['eta'],input['phi'],input['pT'].abs(),0.1,0.1,20,color='y',shade=True)
            surf5 = ax[1,1].bar3d(output['eta'],output['phi'],output['pT'].abs(),0.1,0.1,20,color='b',shade=True)
            for i in [0,1]:
                for j in [0,1]:
                    ax[i,j].set_ylim(-3.14,3.14)
                    ax[i,j].set_xlim(-3.5,3.5)
                    ax[i,j].set_xlabel("phi")
                    ax[i,j].set_ylabel("eta")
                    ax[i,j].set_zlabel("pT")
            ax[1,1].set_xlabel("eta")
            ax[1,1].set_ylabel("phi")
            ax[1,1].set_zlabel("pT")
        
        ax[0,0].legend([in_proxy],['input event'])
        ax[0,1].legend([out_proxy],['AE output event'])
        ax[1,0].legend([in_proxy,out_proxy],['input event','AE output event'])
        ax[1,1].legend([in_proxy,out_proxy],['input event','AE output event'])
        plt.savefig("plots/AE_io_%s_pxpypz.png"%label)
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

input_list = np.loadtxt("test_input_%s.txt"%(ending))[:,:684]
output_list = np.loadtxt("test_output_%s.txt"%(ending))[:,:684]
for i in range(10):
	plot_3D(input_list[i], output_list[i], ending+'_%i'%i)
