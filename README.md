# NGBC
We use the natural neighbor structure to construct coarse-grained granular-balls, and then use granular-balls to implement the clustering method based on ”large-scale priority”.
# Files
These program mainly containing:
  - a  synthetic dataset folder named "syndataset".
  - a  real dataset folder named "dataset".
  - a python file named “Test_Split.py” is the main function file.
  - other python files
# Requirements
## Installation requirements (Python 3.8)
  - Pycharm 
  - Linux operating system or Windows operating system
  - sklearn ,numpy, pandas, scipy
# Dataset Format
  The real dataset should be given in a csv file of the following format:
  - In every line, there are d+1 numbers: the label of the point and the d coordinate values, where the label is an integer from 0 to n-1.
# Usage
  Run the Test_Split.py to test the code on various datasets. The results are viewed by visualizing graphs of the detected clusters in the data for datasets without labels, and by ACC and NMI scores for those with labels.
