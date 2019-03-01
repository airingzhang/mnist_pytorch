'''
Created on Feb 25, 2019

@author: airingzhang
'''
import codecs
import numpy as np
import sys

# Magic number for MNIST dataset
MAGIC_NUM_LABEL = 2049
MAGIC_NUM_IMAGE = 2051

def read_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def process_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert read_int(data[:4]) == MAGIC_NUM_LABEL
        length = read_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return length, parsed 


def process_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert read_int(data[:4]) == MAGIC_NUM_IMAGE
        length = read_int(data[4:8])
        num_rows = read_int(data[8:12])
        num_cols = read_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return length, num_rows, num_cols, parsed 
    
class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
