import h5py
h5f = h5py.File('result_deep_regression_shifted_45_eps','w')
h5f.create_dataset('X', data = X)
h5f.create_dataset('annotated', data = annotated)
h5f.create_dataset('label_GM', data = label)
h5f.close()

h5f = h5py.File('result_deep_regression_shifted_45_eps','r')
predictions_valid_test = h5f['predictions_valid_test'][()]
y_test_ex3 = h5f['y_test_ex3'][()]
h5f.close()
fnames = h5f['filenames'][()]

# Open
# h5f.visit(print) #print all filenames in the file
h5f = h5py.File('X_blasi_original.h5','r')
X = h5f['X'][()]
y = h5f['y'][()]
fnames = h5f['filenames'][()]…h5f.close()


# rsync -a GPU:~/DeepLearningPipeline/result_deep_regression_shifted_45_eps /Users/moritzberthold/Desktop/
# Copy to home
