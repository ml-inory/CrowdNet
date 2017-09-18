# -*- coding: utf-8 -*-
import os
import sys
import matplotlib.pyplot as plt

loss_log = []
loss_iter = []

acc_log = []
acc_iter = []

lr_log = []
lr_iter = []

iter_offset = 0

if len(sys.argv) < 2:
    print 'Usage: python plot_info.py training_info'
    print 'You may specify multiple info files, e.g. python plot_info.py info1 info2 ...'
    sys.exit(1)

if '.jpg' in sys.argv[-1] or '.png' in sys.argv[-1]:
    fig_name = sys.argv[-1]
    info_list = sys.argv[1:-1]
else:
    fig_name = 'loss-acc.png'
    info_list = sys.argv[1:]

for i in xrange(len(info_list)):
    f = open(info_list[i], 'r')
    
    for line in f:
        line = line.strip()
        if 'Iteration' in line and 'loss' in line:
            loss_iter.append(int(line[line.index('Iteration ') + len('Iteration ') : line.index(',')]) + iter_offset)
            loss_log.append(float(line[line.index('loss = ') + len('loss = ') :]))
        if 'Iteration' in line and 'Testing' in line:
            acc_iter.append(int(line[line.index('Iteration ') + len('Iteration ') : line.index(',')]) + iter_offset)
        if 'Test' in line and 'accuracy' in line:
            acc_log.append(float(line[line.index('accuracy = ') + len('accuracy = ') :]))
        if 'Iteration' in line and 'lr' in line:
            lr_iter.append(int(line[line.index('Iteration ') + len('Iteration ') : line.index(',')]) + iter_offset)
            lr_log.append(float(line[line.index('lr = ') + len('lr = ') :]))

    iter_offset += loss_iter[-1]
    f.close()
# print log[-10:]
# print acc_log
loss = plt.plot(loss_iter, loss_log, label='Loss')
#acc = plt.plot(acc_iter, acc_log, label='Accuracy')
plt.legend(loc='best')

plt.savefig(fig_name)
plt.show()    
