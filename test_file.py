import nbformat
import pytest
import numpy as np
# from cnnb.XORNN import *
import re

class TestModel:
    @pytest.fixture(autouse=True)
    def get_model(self):
        file1 = open("cnnb/output.txt","r") 
        # print "Output of Readlines after appending"
        ot = file1.readlines()
        # print(ot)
        
        file1.close()
        temp0 = re.findall(r'\d+\.\d+', ot[0]) 
        self.temp0 = float(temp0[0])
        temp1 = re.findall(r'\d+\.\d+', ot[1]) 
        self.temp1 = float(temp1[0])
        


    def test_nn(self):
        assert 0.7 <= self.temp0 <= 0.8
        assert 0.002 <= self.temp1 <= 0.4