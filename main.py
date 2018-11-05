import pandas as pd
import numpy as np
import numpy.linalg as la
from time import sleep
from random import shuffle
from cvxopt import matrix, solvers
import pickle
import itertools
from tqdm import tqdm
from tqdm import trange



#(0,1) (0,2) (0,3) (0,4) (0,5) (1,2) (1,3) ()

def dataset(file1, file2):
	#data1 = open('./%s.txt' % file1).readlines()
	#data2 = open('./%s.txt' % file2).readlines()
	file_id_1 = open('../label_data/%s.txt' % file1)
	file_id_2 = open('../label_data/%s.txt' % file2)
	data1 = file_id_1.readlines()
	data2 = file_id_2.readlines()
	file_id_1.close()
	file_id_2.close()
	#
	#data1 = ''.join(map(str, data1)).split('\n')
	#data2 = ''.join(map(str, data2)).split('\n')

	array1 = [[str(var) for var in line.split()] for line in data1]
	array2 = [[str(var) for var in line.split()] for line in data2]
	for i in range(len(array2)):
		array1.append(array2[i])

	#print(array1)
	array=[]
	N= len(array1)
	#print(N)
	count=0
	for row in range(N):

		if (int(array1[row][0]) == file1):
			array1[row][0]=-1
		else:
			array1[row][0]=1
		#for col in range(len(row)):

	return (array1)


def linear(x,y):
	z=((np.dot(x,y)))
	if z==0:
		z=-1
	#print(z)
	#sleep(0.5)
	return z

def polynomial(x,y, q=3):
	 return  (1 + np.dot(x, y)) ** q

def gaussian(x,y,sigma=0.05):
	return np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))

def radial_basis(x,y,gamma=0.0005):
	return np.exp(-gamma*la.norm(np.subtract(x, y)))



def sparse(array1):
	outer=[]
	for col in array1:
		outer.append(col.split(':'))
	#N=53000
	N=16
	inner_data=[]
	p=1
	for i in range(len(outer)):
		if(p==int(outer[i][0]) ):
			inner_data.append(outer[i][-1])
			p+=1

		elif (p<=int(outer[i][0])):# and p>int(outer[i-1][0])):
			while(p<int(outer[i][0])):/home/kartik/Desktop/ML_Proj/code/multi_class.py # and p>int(outer[i-1][0])):
				inner_data.append('0')
				p+=1
			if (p==int(outer[i][0])):
				inner_data.append(outer[i][-1])
				p+=1


	#print(inner_data[1:100])
	n=len(inner_data)
	#print(N-n)
	for i in range(N-n):
		inner_data.append('0')

	outer = [0]*N

	
	
	for i in range(	N):
		outer[i]=float(inner_data[i])
	
	#print(outer)
	#for i in range(N):
	#print(outer[1:100])
	return np.array(outer)
	


def disp(X,N):
	for i in range(N):
		print(X[i])



def fit(x, y, NUM):
	K=np.zeros((NUM,NUM))
	for i,xi in enumerate(x):
		for j,xj in enumerate(x):
			X1=np.array(sparse(xi))
			X2=np.array(sparse(xj))
			#sleep(10)
			K[i,j] = gaussian(X1, X2)
		
	
	D=(len(y))
	#y=y.astype(double)
	P= matrix(np.outer(y,y)*K)
	q = matrix(-np.ones((NUM, 1))) #Nx1
	G = matrix(-np.eye(NUM))    # NxN
	h = matrix(np.zeros(NUM))   #Nx1
	#A = matrix(y.reshape(1, -1),'d')
	A=matrix(y, (1,N), 'd') # 1xN
	b = matrix(np.zeros(1))    # 1x1
	
	solvers.options['show_progress'] = False
	sol = solvers.qp(P, q, G, h, A, b)
	alphas = np.array(sol['x'])  #Nx1
	#print(alphas)
	return alphas

def predict(x):
	y=np.dot(w,x)+np.array(bias)
	if( y> 0):
		y1=1
	else:
		y1=-1

	return y1;

def  accuracy(y,x,N):
	n=0
	for i in range(N):
		if(predict(x[i])==y[i]):
			n+=1
	return n/N


def readfile(N):
	pairs = list(itertools.combinations(range(N), 2))
	for i in pairs:
		print(i[0],'\t',i[1])
	sleep(100)
	for f in glob.glob("./data/*.txt"):
		label = (f.split('.')[0]).split('_')


if __name__ == '__main__':
	pairs = list(itertools.combinations(range(10), 2))
	for i in trange(len(pairs)):
		file1, file2 = pairs[i][0],pairs[i][1] 
		train_data = dataset(file1,file2)
		N=len(train_data)
		y=np.array([train_data[i][0] for i in range(N)])
		x=np.array([train_data[i][1:] for i in range(N)])

		print('data readfile: %s %s ' %(file1, file2))

		alphas = fit(x, y, N)
		Input=[]
		for i in range(len(x)):
			Input.append(sparse(x[i]))

		Input = np.array(Input)

		w = np.sum(alphas * y[:, None] * Input, axis = 0)
		cond = (alphas > 1e-4).reshape(-1)
		b = y[cond] - np.dot(Input[cond], w)	
		
		norm = np.linalg.norm(w)
		bias=0
		if (len(b)>0):
			bias = b[0]
			bias = bias/norm
		
		w, bias = w / norm, bias / norm

		with open('../gaussian_1/%s_%s.pickle' %(file1, file2), 'wb') as f:
			pickle.dump((w, bias), f)
		print('%2f' %(accuracy(y,Input,len(Input))*100),'%', '	files:  ',(file1, file2))
