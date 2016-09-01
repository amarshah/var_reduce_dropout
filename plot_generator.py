# code to take generated data from mnist experiments and plot it

import numpy as np
import cPickle
import matplotlib.pyplot as plt

# save_file = "mirror.pkl"

# with open(save_file, "rb") as f:
# 	data = cPickle.load(f)

# test_stoch_acc = np.array(data["test_losses_stoch"])
# test_non_stoch_acc = np.array(data["test_losses_non_stoch"])

# x = np.arange(test_stoch_acc.shape[-1])

# plt.plot(x, test_stoch_acc.mean(axis=0), 'r',
# 	     x, test_non_stoch_acc.mean(axis=0), 'b')

# plt.show()

###############################################################

save_files = ["mirror.pkl", "dropout.pkl", "dropout2.pkl", "nodropout.pkl"]

results = []
errors = []
n_results = []
for save_file in save_files:
	with open(save_file, "rb") as f:
		data = cPickle.load(f)
	acc = np.array(data["test_losses_non_stoch"])
	results.append(acc.mean(axis=0))
	errors.append(acc.std(axis=0))
	n_results.append(acc.shape[-1])
	print acc.shape[0]

# import pdb
# pdb.set_trace()

fig = plt.figure()
labels = ["mirror", "1 mask", "2 masks", "0 masks"]
colors = ["red", "blue", "green", "black"]
for i in xrange(len(results)):
	plt.errorbar(np.arange(n_results[i])[6:], results[i][6:],
		         xerr=errors[i][6:], color=colors[i], label=labels[i])
plt.legend(loc=4)

# plt.plot(np.arange(n_results[0]), results[0], 'r',
# 	     np.arange(n_results[1]), results[1], 'b',
# 	     np.arange(n_results[2]), results[2], 'g',
# 	     np.arange(n_results[3]), results[3], 'k')



plt.show()


