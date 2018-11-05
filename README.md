THIS FILE CONTAIN WEIGHT VALUE OF SVM ON RADIAL BASIS.

Requirment:
	pickles
	numpy
	glob
	time

TO TEST ON MODEL:
	RUN: python3 multiclass.py 

TO TEST ON PATRICULAR CLASS ,
	uncomment line 8 and  comment line 9
    RUN: python3 multiclass.py n   ; where n=(0,1,.....9)

We have provided a processed data where for each class label contained in corresponding class label.txt

We have also implemented RANDOM FOREST AND NAIVE BAYES contain in RF.py and naivebayes.py 


For Cross checking of our implemention
        Do following:
                Data preprocess:
                        run: python3 data_partition.py
                             python3 main.py 
                             
