# Neural-Network-from-scratch
Implementation of Neural Network on LBW_Dataset from scratch.

Low Birth weight (LBW) acts as an indicator of sickness in newborn babies. LBW is closely
associated with infant mortality as well as various health outcomes later in life. Various studies
show strong correlation between maternal health during pregnancy and the child’s birth weight.
We use health indicators of pregnant women such as age, height, weight, community etc in order
for early detection of potential LBW cases. This detection is treated as a classification problem
between LBW and not-LBW classes

The Dataset consists of 10 columns\
1. Community - Information about the patient’s community\
  a. SC - 1\
  b. ST - 2\
  c. BC - 3\
  d. OC - 4\
2. Age - Patients age in years\
3. Weight - Weight in Kg during Trimester\
4. Delivery Phase -\
  a. 1 - Before 37 weeks\
  b. 2 - After 42 weeks\
5. HB - Haemoglobin content\
6. IFA - determines if the patient took Folic acid or not\
a. 1 - patient consumed Folic acid\
b. 0 - patient did not consume Folic acid\
7. BP - Blood Pressure during Trimester\
8. Education - Educational Qualification of the patient on a scale of 0-10\
9. Residence - indicates whether the patient is resident of the town or village in which the\
study was conducted (indicated by 1) or if the patient is a non-resident (indicated by 2)\
10. Result - Label 1 indicates case of LBW, Label 0 indicates non LBW case\

'''
Description of neural network implemented and hyperparameters used:

	step 1: Pre-processing
		This includes both cleaning and transforming.
		All the missing values are filled by mean\mode\groupby_and_then_mean as per the requirement.
		(for eg: the missing values of 'weight' column are filled as, first grouping them by 'age' and then taking mean)
		After cleaning, dataset is standardized for numerical values
		for categorical, one-hot-encoding is used.

	step 2: Implementing Neural Network	
		1.Intializing Weight and Bias Matrices:
			Convention followed: weight matrix of layer 'L' has dimension (NL,NL-1) 
								Bias matrix of Layer 'L' has dimension (NL,1)
								(Note: 'NL' indicates no of hidden units in layer 'L' similarly,
								'NL-1' indicates no of hidden units in layer 'L-1')
		2.Forward Propogation:
			There are 5 layers in the neural network including the first input layer.
			No of hidden units in each layer:
				1.Input Layer: No of features in the dataset (i.e 12 units)
				2.First Hidden Layer: 15 units
				3.Second Hidden Layer: 5 units
				4.Third Hidden Layer: 2 units
				5.Output Layer: 1 unit
			Activation functions used:
				Output Layer has 'Sigmoid' Activation Function.
				Except Output Layer, all other layers have 'Relu' as their activation function.
		3.Backward Propogation:
			Loss Function: L(yhat,y)=-(y*log(yhat) + ((1-y)*log(1-yhat))) with gradient descent is used.
						where 'y' is true value and 'yhat' is predicted value.                       
			Learning Rate: 0.01
			No of Iterations: 9000

Key Features:

	1.Pre-processing: Pre-processing of data is extremly important as right ways of cleaning and transforming leads to the
					  desired goal quickly with minimum amount of effort. So we made sure that pre-processing is done much more 
					  efficently as described above.
					  (for eg: Normalizing/Standardizing numerical inputs and one-hot-encoding for categorical input features,
					  which makes neural network to be more efficent.)
					  
	2.Loss Function: Instead of MSE as loss function, logistic regression loss function is used which works similar 
					 to MSE, but is efficent than MSE (i.e converging to global optimum when there are multiple local optimums)
					 
	3.Intializing Weights and Biases: Careful choice of initization has to be made to partially solve the problem of 
									  exploding/vanishing gradients.So the random weights are multiplied by sqrt(2/NL-1),
									  to scale them such that their mean and standard deviation are approx 0 and 1 respectively.
									  
	4.Good Choice of Hyper-parameters: Tuning of Hyper-parameters is also essential in-order to obtain better results,
									   Combination of Hyper-parameters which gave us better performance are:
									   1.Sigmoid activation function at output layer and Relu at rest all.
									   2.Four Layers with 15,5,2,1 hidden units in respective layers.
									   2.learning-rate: 0.01
									   3.No of Iterations: 9000
									   with the results, Training Accuracy of 98.5% and Test Accuracy of 86.2%
 Steps to run "main.py" file:
 
	(Go to "src" folder i.e 
	> cd src) and then,
	> python main.py
	
	(versions: python->3.8,pandas->1.0.4,numpy->1.18.4,sklearn->0.23.2)
	(Note: to run "pre_processing_source_code.py"
		> python pre_processing_source_code.py)
		
'''
