#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import pylab as plt
import plotly.plotly as py



#fname = sys.argv[1]
fname = 'pred_vs_true.dat'

f = open(fname)
pred, true = [], []
skip = 1
for i, line in enumerate(f):
	if i % skip == 0:
		pred.append(float(line.split()[0]))
		true.append(float(line.split()[1]))

pred = np.array(pred)
true = np.array(true)

print(pred)
print(true)



fig1 = plt.figure()
ax = fig1.add_subplot(2, 1, 1)
ax.scatter(pred, true, color = 'b')

fig2 = plt.figure()
ax1 = fig2.add_subplot(2, 1, 1)
n, bins, patches = plt.hist([pred])


fname2 = 'pred_vs_true2.dat'

f = open(fname2)
pred, true = [], []
skip = 1
for i, line in enumerate(f):
	if i % skip == 0:
		pred.append(float(line.split()[0]))
		true.append(float(line.split()[1]))

pred = np.array(pred)
true = np.array(true)

print(pred)
print(true)

ax = fig1.add_subplot(2, 1, 2)
ax.scatter(pred, true, color = 'g')


ax1 = fig2.add_subplot(2, 1, 2)
n, bins, patches = plt.hist([pred])
'''
color_distro = np.linspace(np.min(abs(true- pred)),np.max(abs(true- pred)), 100)
h = (np.max(abs(true - pred)) - np.min(abs(true- pred))) / (100 - 1)
colors = np.empty(len(true))
x = np.linspace(np.min(true), np.max(true), 100)

#for i, (t, p) in enumerate(zip(true, pred)):
#	colors[i] = color_distro[int(abs(t - p) / h)]
	
fig = plt.figure()
ax = fig.gca()

plt.scatter(true, pred, s=10, cmap='rainbow')
plt.plot(x,x, '--', color='k', markersize=20)
#plt.colorbar()
plt.grid()


# plt.title('True vs. predicted distances for $H_2$', {'family': 'serif','fontsize': 15})
plt.xlabel('True distance [$\AA$]', {'family': 'serif','fontsize': 12})
plt.ylabel('Predicted distance [$\AA$]', {'family': 'serif','fontsize': 12})

ax.set_xlim([round(np.min(true)), round(np.max(true))])
ax.set_ylim([round(np.min(pred)), round(np.max(pred))])

ticklines = ax.get_xticklines() + ax.get_yticklines()
gridlines = ax.get_xgridlines() + ax.get_ygridlines()
ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

for line in ticklines:
    line.set_linewidth(3)

for line in gridlines:
    line.set_linestyle('-.')

for label in ticklabels:
    # label.set_color('r')
    label.set_fontsize('medium')
'''
plt.show()
plt.show()
