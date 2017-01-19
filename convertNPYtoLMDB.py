import numpy as np
import lmdb
import caffe
import code

filename = "all_channels_80_80_full_no_zeros_in_cells"
extension = ".npy"
path = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex2/PreparedData/"
labels = np.load("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex2/PreparedData/labels_80_80_full_no_zeros_in_cells.npy")
X = np.load(path+filename+extension)
print "MAX", np.max(X)
code.interact(local=dict(globals(), **locals()))


def convertNPYtoLMDB(X, labels, path, db_name):
	N = X.shape[0]
	# X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
	# y = np.zeros(N, dtype=np.int64)
	save_as = path + db_name
	map_size = X.nbytes * 5
	env = lmdb.open(save_as, map_size=map_size)
	with env.begin(write=True) as txn:
	    # txn is a Transaction object
	    for i in range(N):
	        datum = caffe.proto.caffe_pb2.Datum()
	        datum.channels = X.shape[1]
	        datum.height = X.shape[2]
	        datum.width = X.shape[3]
	        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
	        datum.label = int(labels[i][0])
	        str_id = '{:08}'.format(i)
	        # The encode is only essential in Python 3
	        txn.put(str_id.encode('ascii'), datum.SerializeToString())



convertNPYtoLMDB(X, labels, path, filename)


