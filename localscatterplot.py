from data_tools import *
# from algorithms import *
from plot_lib import *
import numpy as np
from scipy.stats import pearsonr, spearmanr
import code
import os
import h5py
from scipy.stats import pearsonr, spearmanr


h5f = h5py.File('/Users/moritzberthold/Desktop/shifted/data/result_deep_regression_shifted_45_eps_cm_cd','r')
# acc = .9490 #dd
acc = .9451 #cc
# acc = .9463 #cd
# acc = .8999 #dc

ch = "RIIb"
threshold = 1000

predictions_valid_test = h5f['predictions_valid_test'][()]
y_test_ex3 = h5f['y_test_ex3'][()]
h5f.close()

# code.interact(local=dict(globals(), **locals()))
from statsmodels.stats.proportion import proportion_confint
cf=proportion_confint(acc*len(y_test_ex3),len(y_test_ex3),method="wilson")
acc_low=cf[0]
acc_high=cf[1]

print("acc_low ", acc_low)
print("acc_high ", acc_high)

# i = np.argsort(y_test_ex3)
# y_test_ex3 = y_test_ex3[i]
# predictions_valid_test = predictions_valid_test[i]
# plt.plot(predictions_valid_test,c='r',alpha=0.5)
# plt.plot(y_test_ex3,c='blue',alpha=0.5, linewidth=3)
# plt.xlim([0, len(y_test_ex3)])
# plt.ylim([0, 6000])
# plt.title(str(ch) + " intensity estimate and ground truth ordered by magnitude")
# plt.ylabel('Intensity')
# plt.xlabel('Cells ordered by ground truth intensity magnitude')
# plt.show()

spearman = round(spearmanr(y_test_ex3, predictions_valid_test)[0],2)
pearson = round(pearsonr(y_test_ex3, predictions_valid_test)[0],2)
rms3 = round(rms(y_test_ex3, predictions_valid_test)**.5,2)


print("Discarding R2b >= ", threshold)
y_test_ex3_ = y_test_ex3[y_test_ex3 < threshold]
# code.interact(local=dict(globals(), **locals()))
predictions_valid_test_ = predictions_valid_test[y_test_ex3 < threshold]

rms3_t = round(rms(y_test_ex3_, predictions_valid_test_)**.5,2)




plt.scatter(predictions_valid_test, y_test_ex3, marker='.', color='r', s = 2, label="Pearson correlation = " + str(pearson) + \
" \nSpearman correlation = " + str(spearman) + \
" \nRMS test = " + str(rms3) +\
" \nRMS test (RIIb < "+ str(threshold) + ")= " + str(rms3_t))
plt.xlim([0, 4000])
plt.ylim([0, 4000])
plt.ylabel('Prediction')
plt.xlabel('Ground Truth')
plt.title("Intensity prediction of channel " + str(ch) + " using Random Forest regression")
leg = plt.legend(loc="upper right")
for item in leg.legendHandles:
    item.set_visible(False)
plt.show()


# plotBothConfusionMatrices(pred_label, labels_test_ex3, class_names)
# code.interact(local=dict(globals(), **locals()))
