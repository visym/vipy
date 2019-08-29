import numpy as np
import cv2
import cv2.cv as cv


def randn(m,n):
    return np.random.randn(m,n)

def rand(m,n):
    return np.random.random( (m,n) )

def uniform_random_in_range(rng=(0,1)):
    return (rng[1] - rng[0]) * np.random.random_sample() + rng[0]

