import scipy.io
import pickle

path = 'pickle\CMU\male\Part 1\seq1-body0_point_cloud_1.mat'
#path = 'CMU\male\Part 1\seq1-body0.dat'
mat = scipy.io.loadmat(path)
"""with open(path, 'rb') as fin :
    a = pickle.load(fin)"""
