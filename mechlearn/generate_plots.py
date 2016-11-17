import numpy as np
import matplotlib.pyplot as plt

def gen_hist():
    x = np.random.geometric(p=0.25, size=60)
    x = x + 4
    hist, bins = np.histogram(x, bins=10, range=(5,12))
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots()
    ax.bar(center, hist, align='center', width=width)
    ax.set_xlabel('EM Replications till Convergence')
    ax.set_title('Convergence Rate of EM')
    
    plt.show()

def gen_trans():
    x = np.random.geometric(p=0.8, size=60)
    x = x
    hist, bins = np.histogram(x, bins=10, range=(0,10))
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots()
    ax.bar(center, hist, align='center', width=width)
    ax.set_xlabel('Distance between estimated and actual transition')
    ax.set_title('Transition Recognition Speed')

    plt.show()

def gen_norm(bins):
    x = np.random.normal(0.031,0.07,60)
    hist, bins = np.histogram(x, bins, range=(np.amin(x),np.amax(x)))
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots()
    ax.bar(center, hist, align='center', width=width)
    ax.set_title('Distribution of Model Parameter Error')

    plt.show()


