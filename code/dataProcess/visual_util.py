import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import pandas as pd
import torch, random

from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d

# according to the data, the final real number may be var.
nc_dic = {'A':0, 'T':1, 'G':2, 'C':3, 'N':4}
cn_dic = {'0':'A', '1':'T', '2':'G','3':'C', '4':'N'}

# visulaization segments
def vis_segment(signal, events, basecall):

	start, end = 0, 40
	bks = np.cumsum([e[1] for e in events[start:end]])
	bks = [0] + bks.tolist()

	bases = [e[2] for e in events[start:end]]
	signal = signal[:bks[-1]+1]
	fig=plt.figure(figsize=(10,6))

	plt.subplot(211)
	plt.plot(range(len(signal)), signal)
	plt.xticks(bks[:-1], bases, color="brown",fontsize=10)
	for bk in bks[:-1]:
		plt.axvline(bk, linestyle="-.", color="red")

	plt.subplot(212)
	base_quality = [ord(s)-33 for s in basecall[3]]
	plt.plot(range(len(base_quality)), base_quality)

	plt.show()
	plt.close("all") 

def plot_curve(train_loss, validate_loss, model_name, save_path):

	plt.plot(train_loss)
	plt.plot(validate_loss)

	plt.title(' %s training curve' %(model_name))
	plt.ylabel('cross entorpty loss')
	plt.xlabel('epoch')
	plt.legend(['train loss', 'test loss'], loc='upper right')

	plt.savefig(save_path)

def plot_curve_GAN(train_loss, model_name, save_path):

	plt.plot([x[0] for x in train_loss])
	plt.plot([x[1] for x in train_loss])

	plt.title(' %s training curve' %(model_name))
	plt.ylabel('cross entorpty loss')
	plt.xlabel('epoch')
	plt.legend(['Generator Loss', 'Discriminator Loss'], loc='upper right')

	plt.savefig(save_path)


def plot_seq_signal_diff(s, meth, unmeth, save_path):

	if len(meth) <= 1 or len(unmeth) <= 1:
		return -1

	fig = plt.figure(figsize=(10,5))
	ax1 = fig.add_subplot(2, 1, 1)
	ax1.boxplot(meth)
	ax1.set_title('Methyl_signal')
	ax1.set_ylabel('pA')
	ax1.set_xlabel('position')

	ax2 = fig.add_subplot(2, 1, 2)
	ax2.boxplot(unmeth)
	ax2.set_title('unMethyl_signal')
	ax2.set_ylabel('pA')
	ax2.set_xlabel('position')

	plt.subplots_adjust(wspace=0.1, hspace=0.6)
	plt.title(' %s signals difference' %("".join([cn_dic[str(x)] for x in s])))
	plt.savefig(save_path + s + ".png")


def plot_seq_signal_diff_group(s, meth, unmeth, save_path):

	if len(meth) <= 1 or len(unmeth) <= 1:
		return -1

	nLen = meth.shape[1]
	column_names = [str(x+1) for x in range(nLen)]
	unmeth = pd.DataFrame(unmeth, columns=column_names).assign(s_type="unMeth")
	meth   = pd.DataFrame(meth,   columns=column_names).assign(s_type="Meth")
	combined = pd.concat([unmeth, meth])

	melted = pd.melt(combined, id_vars="s_type")

	fig = plt.figure(figsize=(10,5))
	sns.boxplot(x="variable", y="value", hue="s_type", order=column_names,palette=["g", "r"], data=melted)

	plt.title(' %s signals difference' %("".join([cn_dic[str(x)] for x in s])))
	plt.savefig(save_path + s + ".png")	


# 20200608
def vis_signal_difference(data_generator, figSavePath="../experiment/figures/barplot.png"):

	meth_list, unMeth_list, sim_list = [],[],[]

	for i, data in enumerate(data_generator, 0):
		
		inputs, labels = data
		if len(labels) == 0: continue

		index1 = labels.nonzero()
		index0 = (labels == 0).nonzero()
		
		if len(index1) > 0:
			meth_list.append(inputs[2][index1,:,0].squeeze(1))

		if len(index0) > 0:
			unMeth_list.append(inputs[2][index0,:,0].squeeze(1))

		sim_list.append(inputs[3][:,:,0])

	# cat the data
	meth_df = torch.cat(meth_list, 0).cpu().numpy()
	unMeth_df = torch.cat(unMeth_list, 0).cpu().numpy()
	sim_df = torch.cat(sim_list, 0).cpu().numpy()

	# visualization
	fig = plt.figure(figsize=(5,10))
	ax1 = fig.add_subplot(3, 1, 1)
	ax1.boxplot(meth_df)
	#ax1.set_ylim([-5,5])
	ax1.set_title('Methylation')

	ax2 = fig.add_subplot(3, 1, 2)
	ax2.boxplot(unMeth_df)
	#ax2.set_ylim([-5,5])
	ax2.set_title('un_Methylation')

	ax3 = fig.add_subplot(3, 1, 3)
	ax3.boxplot(sim_df)
	#ax3.set_ylim([-5,5])
	ax3.set_title('Simulation data')

	plt.subplots_adjust(wspace=0.1, hspace=0.6)
	plt.savefig(figSavePath)


# visulaization for sequence contentnt
def tSNE_plot(X, Y, file_save_path, max_num=-1, dim=2):

	if max_num > 0 and max_num < len(Y):
		sidx = random.sample(range(len(Y)),max_num)
		X = X[sidx]
		Y = Y[sidx]

	labels = [0, 1]
	colors = ["blue", "red"]
	plt.figure(figsize=(10,10))

	if dim == 3:
		ax = plt.axes(projection='3d')

	latent_vec = TSNE(n_components=dim, random_state=0).fit_transform(X)
	for i in range(len(labels)):
		idx = np.where(Y == labels[i])[0]
		
		if dim == 2:
			plt.scatter(latent_vec[idx, 0], latent_vec[idx, 1], c=colors[i])
		elif dim == 3:
			ax.scatter3D(latent_vec[idx, 0], latent_vec[idx, 1],latent_vec[idx, 2], c=colors[i])

	plt.legend(["unMeth", "Meth"])
	plt.savefig(file_save_path)

	plt.clf()
	plt.close()








		



