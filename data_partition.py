import numpy as np
import os
file=open('../data/test.txt')
 
for ver in file:
	name=(ver.strip().split()[0])
	#print(name)
	os.system('mkdir ../label_data_test/')
	doc=open('../label_data_test/%s.txt' % name, 'a')	
	doc.write(ver)
	doc.close()
'''
for i in range(N):
	print(data[i][0])

	doc=open('./parse/%s.txt' % data[i][0:1], 'a')
	doc.write(data[i][:])
	doc.close()
'''

