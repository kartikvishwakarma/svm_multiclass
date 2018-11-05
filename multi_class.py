import glob, os
import sys
import pickle
import numpy as np
from time import sleep

file = (sys.argv)[1]
#data_file = open('../label_data_test/%s.txt' % file)
data_file = open('../data/test.txt')
data1 = data_file.readlines()
test = [[str(var) for var in line.split()] for line in data1]
os.chdir("../radial_basis/")
N=16
def sparse(array1):
	outer = []
	
	for col in array1:
		outer.append(col.split(':'))
	
	#N=53000
	#print(outer)
	#sleep(100)
	inner_data=[]
	p=1
	for i in range(len(outer)):
		#print('%s %s' %(p , outer[i][0]))
		if(p==int(outer[i][0]) ):
			inner_data.append(outer[i][-1])
			p+=1

		elif (p<=int(outer[i][0])  and p>int(outer[i-1][0])):
			while(p<int(outer[i][0]) and p>int(outer[i-1][0])):
				inner_data.append('0')
				p+=1
			if (p==int(outer[i][0])):
				inner_data.append(outer[i][-1])
				p+=1

	n=len(inner_data)

	for i in range(N-n):
		inner_data.append('0')

	outer = [0]*N
	
	for i in range(	N):
		outer[i]=float(inner_data[i])
	
	#print(np.array(outer))
	#sleep(3)
	return np.array(outer)


margin=[]
def predict(x):
	feq = [0]*N
	for f in glob.glob("*.pickle"):
			label = (f.split('.')[0]).split('_')
			with open(f, 'rb') as f:
				w,bias = pickle.load(f)
				y1= np.dot(w,x)+bias
				if(y1>0):
					feq[int(label[1])]+=1.0 
				else:
					feq[int(label[0])]+=1.0
	#print(feq.index(max(feq)))
	return (feq.index(max(feq)))
	



if __name__ == '__main__':
	
	count=0
	N1=len(test)
	for i in range(N1):
		#print('%s %s '%( test[i][0], test[i][1:] ))
		#sleep(1)
		if(int(test[i][0]) == predict(sparse(test[i]))): 		
			count+=1
	print(count/N1)














