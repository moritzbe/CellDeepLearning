import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import code
# path = "~/dl/cellmodels/blasi/120617/_train_log.csv"
path = "~/dl/cellmodels/deepflow/120617/predict_r2b_intensity_based_on_PGP_no_bleed_trough_shifted_no_bs_drop=3.csv.csv"
# path = "~/dl/cellmodels/deepflow/120617/predict_r2b_intensity_based_on_PGP_no_bleed_trough_shifted.csv"
# path = "~/dl/cellmodels/deepflow/120617/predict_r2b_intensity_based_on_PGP_no_bleed_trough_shifted_no_bs.csv"
tb = pd.read_table(path, delimiter=",")

loss = tb["loss"]
val_loss = tb["val_loss"]
code.interact(local=dict(globals(), **locals()))
plt.plot(loss,c='r',alpha=0.5, linewidth=3)
plt.plot(val_loss,c='blue',alpha=0.5, linewidth=3)
plt.xlim([0, loss.size])
plt.ylim([0, 400000])
np.max(loss)
# plt.ylim([0, np.max([np.max(loss), np.max(val_loss)])])
plt.title("Learning curves, train and val")
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Training epochs')
plt.show()



acc = tb["acc"]
val_acc = tb["val_acc"]




plt.plot(acc,c='r',alpha=0.5, linewidth=3)
plt.plot(val_acc,c='blue',alpha=0.5, linewidth=3)
plt.xlim([0, acc.size])
plt.ylim([0, 1])
# plt.ylim([0, np.max([np.max(loss), np.max(val_loss)])])
plt.title("Learning curves, train and val accuracies")
plt.ylabel('Accuracy')
plt.xlabel('Training epochs')
plt.show()
