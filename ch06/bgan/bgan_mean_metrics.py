import matplotlib.pyplot as plt
import csv
import numpy as np

x = []
y_d_loss = []
y_d_accuracy = []
y_g_loss = []

with open('bgan_metrics.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y_d_loss.append(float(row[1]))
        y_d_accuracy.append(float(row[2]))
        y_g_loss.append(float(row[3]))

mean_d_loss = np.mean(y_d_loss)
mean_d_accuracy= np.mean(y_d_accuracy)
mean_g_loss = np.mean(y_g_loss)

print('mean d loss:' + str(mean_d_loss))
print('mean g loss:' + str(mean_g_loss))
print('mean d accuracy:' + str(mean_d_accuracy))