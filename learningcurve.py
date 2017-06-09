import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "_train_log.csv"
tb = pd.read_table(path, delimiter=",")

loss = tb["loss"]
val_loss = tb["val_loss"]
acc = tb["acc"]
val_acc = tb["val_acc"]


#
plt.plot(loss,c='r',alpha=0.5, linewidth=3)
plt.plot(val_loss,c='blue',alpha=0.5, linewidth=3)
plt.xlim([0, loss.size])
plt.ylim([0, 2])
# plt.ylim([0, np.max([np.max(loss), np.max(val_loss)])])
plt.title(str(ch) + "Learning curves, train and val")
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Training epochs')
plt.show()


plt.plot(acc,c='r',alpha=0.5, linewidth=3)
plt.plot(val_acc,c='blue',alpha=0.5, linewidth=3)
plt.xlim([0, acc.size])
plt.ylim([0, 1])
# plt.ylim([0, np.max([np.max(loss), np.max(val_loss)])])
plt.title(str(ch) + "Learning curves, train and val accuracies")
plt.ylabel('Accuracy')
plt.xlabel('Training epochs')
plt.show()
