import matplotlib.pyplot as plt
import csv
import numpy as np

x = []
y_d_loss = []
y_d_accuracy = []
y_g_loss = []
with open('gan_metrics.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y_d_loss.append(float(row[1]))
        y_d_accuracy.append(float(row[2]))
        y_g_loss.append(float(row[3]))

#
# plt.plot(x,y, label='Accuracy')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('GAN Metrics\n Accuracy Plot')
# plt.legend()
#
# #plt.save('d-loss.png')
# fig = plt.figure()
# plt.savefig('accuracy-plot.png', dpi=fig.dpi)
# plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Simple data to display in various forms
#x = np.linspace(0, 2 * np.pi, 400)
#y = np.sin(x ** 2)



# Just a figure and one subplot
#f, ax = plt.subplots()
#ax.plot(x, y)
#ax.set_title('Simple plot')

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(3, sharex=True)

axarr[0].plot(x, y_d_loss)
axarr[0].set_title('GAN Discriminator Loss')

axarr[1].plot(x, y_d_accuracy,color='r')
axarr[1].set_title('GAN Discriminator Accuracy')
axarr[2].plot(x, y_g_loss,color='m')
axarr[2].set_title('GAN Generator Loss')
plt.subplots_adjust(left=0.11, bottom=0.1, right=0.90, top=0.90, wspace=0.0, hspace=0.4)

plt.show()