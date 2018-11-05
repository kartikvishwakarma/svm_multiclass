from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np   
import pandas as pd 
from time import sleep	
def Read_data():
	file = open('../data/train.txt')
	data = file.readlines()
	test_data = [[str(var) for var in line.split()] for line in data]
	return test_data
	
def sparse(array1):
	outer=[]
	for col in array1:
		outer.append(col.split(':'))
	
	inner_data=[]
	p=1
	for i in range(len(outer)):
		#print('%s   %s' %(p ,outer[i][0]))
		#sleep(1)
		if(p==int(outer[i][0]) ):
			inner_data.append(outer[i][-1])
			p+=1

		elif (p<=int(outer[i][0]) ):# and p>int(outer[i-1][0])):
			while(p<int(outer[i][0]) ):# and p>int(outer[i-1][0])):
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
	#sleep(1)
	return np.array(outer)


if __name__ == '__main__':
	test_data=Read_data()
	N=16
	N1=len(test_data)
	#X=np.zeros((N,D))
	y=np.array([test_data[i][0] for i in range(N1)])
	x=np.array([test_data[i][1:] for i in range(N1)])
	X=[]
	for i in range(N1):
		X.append(np.array(sparse(x[i])))
	
	X=np.array(X)
	#print(X[0:10])
	
	clf = GaussianNB()
	clf.fit(X,y)
	tmp=clf.predict(X)
	print(tmp)
	count=0
	for i in range(len(tmp)):
		print('%s %s' %(tmp[i], y[i]))
		if tmp[i] == y[i]:
			count = count+1
		#sleep(1)
	print (float(count/i)*100)
	
