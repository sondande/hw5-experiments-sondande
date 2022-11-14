[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=9329216&assignment_repo_type=AssignmentRepo)
# hw5-experiments
HW5: Experiments and Technical Writing

1.	monks1.csv: A data set describing two classes of robots using all nominal attributes and a binary label.  This data set has a simple rule set for determining the label: if head_shape = body_shape OR jacket_color = red, then yes, else no. Each of the attributes in the monks1 data set are nominal.  Monks1 was one of the first machine learning challenge problems (http://www.mli.gmu.edu/papers/91-95/91-28.pdf).  This data set comes from the UCI Machine Learning Repository: http://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems

2.	mnist_100.csv: A data set of optical character recognition of numeric digits from images.  Each instance represents a different grayscale 28x28 pixel image of a handwritten numeric digit (from 0 through 9).  The attributes are the intensity values of the 784 pixels. Each attribute is ordinal (treat them as continuous for the purpose of this assignment) and a nominal label.  This version of MNIST contains 100 instances of each handwritten numeric digit, randomly sampled from the original training data for MNIST.  The overall MNIST data set is one of the main benchmarks in machine learning: http://yann.lecun.com/exdb/mnist/.  It was converted to CSV file using the python code provided at: https://quickgrid.blogspot.com/2017/05/Converting-MNIST-Handwritten-Digits-Dataset-into-CSV-with-Sorting-and-Extracting-Labels-and-Features-into-Different-CSV-using-Python.html

3.	votes.csv: A data set describing the voting histories of members of the U.S. Congress.  The objective is to predict a member’s political party based on their previous votes.  All attributes are nominal, although some values are missing (indicated by a “?” character).  This data comes from Weka 3.8: http://www.cs.waikato.ac.nz/ml/weka/

4.	hypothyroid.csv: A data set describing patient health data using a mix of nominal and continuous attributes that can be used to diagnose the health of a patient’s thyroid into four possible labels. This data set comes from Weka 3.8: http://www.cs.waikato.ac.nz/ml/weka/
