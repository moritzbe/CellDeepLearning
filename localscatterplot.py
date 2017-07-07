from data_tools import *
# from algorithms import *
from plot_lib import *
import numpy as np
from scipy.stats import pearsonr, spearmanr
import code
import os
import h5py

h5f = h5py.File('/Users/moritzberthold/Desktop/shifted/data/result_deep_regression_shifted_45_eps_dm_dd','r')
predictions_valid_test = h5f['predictions_valid_test'][()]
y_test_ex3 = h5f['y_test_ex3'][()]
h5f.close()

code.interact(local=dict(globals(), **locals()))

i = np.argsort(y_test_ex3)
y_test_ex3 = y_test_ex3[i]
predictions_valid_test = predictions_valid_test[i]
plt.plot(predictions_valid_test,c='r',alpha=0.5)
plt.plot(y_test_ex3,c='blue',alpha=0.5, linewidth=3)
plt.xlim([0, len(y_test_ex3)])
plt.ylim([0, 6000])
plt.title(str(ch) + " intensity estimate and ground truth ordered by magnitude")
plt.ylabel('Intensity')
plt.xlabel('Cells ordered by ground truth intensity magnitude')
plt.show()


# plotBothConfusionMatrices(pred_label, labels_test_ex3, class_names)
