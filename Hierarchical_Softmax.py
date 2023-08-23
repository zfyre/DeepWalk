import torch
import random


def next_2_pow(num):
    if(num&(num-1) == 0):
        return num
    else:
        cnt = 0
        while(num!=0):
            num//=2
            cnt+=1
        return (1<<cnt)

def BuildRand(vertices):
    n = len(vertices)
    m = next_2_pow(n)

            

def BuildZipf():
    pass

def Build(vertices,zipfs_dist=None):
    if(zipfs_dist == None):
        return Build_Zipf(vertices,zipfs_dist)
    else:
        return Build_rand(vertices)