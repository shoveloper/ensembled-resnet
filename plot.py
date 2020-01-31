import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('log1', type=str, help='')
parser.add_argument('log2', type=str, help='')
args = parser.parse_args()

log_file1 = open(args.log1, 'r')
log_file2 = open(args.log2, 'r')

train_acc1 = []
valid_acc1 = []
for line in log_file1.readlines():
    tokens = line.strip().split()
    if len(tokens) >= 16 and tokens[14] == '/':
        train_acc1.append(float(tokens[13]))
        valid_acc1.append(float(tokens[15]))

train_acc2 = []
valid_acc2 = []
for line in log_file2.readlines():
    tokens = line.strip().split()
    if len(tokens) >= 16 and tokens[14] == '/':
        train_acc2.append(float(tokens[13]))
        valid_acc2.append(float(tokens[15]))

x_axis = list(np.arange(1, 201))
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(x_axis, train_acc1, 'tab:green', label='Train Acc. (resnet18)')
plt.plot(x_axis, valid_acc1, 'tab:blue', label='Valid Acc. (resnet18)')
plt.plot(x_axis, train_acc2, 'tab:orange', label='Train Acc. (proposed method)')
plt.plot(x_axis, valid_acc2, 'tab:red', label='Valid Acc. (proposed method)')
plt.title('Train & Valid Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy', labelpad=0.1)
plt.legend()
plt.show()