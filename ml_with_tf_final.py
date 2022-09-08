import numpy as np 
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tensorflow.keras.layers import Dense, Dropout,ReLU,LayerNormalization,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from keras import backend as K
# define some helpful functions
# def create_Morgan_fp(mols,depth = 1,bitnum = bits):
# 	fps = [AllChem.GetHashedMorganFingerprint(m, depth, nBits=bitnum,useFeatures=False) for m in mols]
# 	nats = [m.GetNumAtoms() for m in mols]
# 	fps_np = np.zeros((len(fps),bitnum+1),dtype=int)
# 	for mol,fp in enumerate(fps):
# 		array = np.zeros((0,), dtype=int)
# 		DataStructs.ConvertToNumpyArray(fp, array)
# 		array=np.append(array,nats[mol])
# 		fps_np[mol]=array
# 	fp_places=np.arange(bitnum,dtype=int)
# 	return fps_np
opt=tf.keras.optimizers.Adam(learning_rate=0.01)

# rewrite the pearson function using tf, so that it could be output in every epoch
def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

# original pearson function to calculate pearson coefficient
def pearson(n_x,n_y):
	a_x = np.average(n_x)
	a_y = np.average(n_y)
	x = n_x - a_x
	y = n_y - a_y
	return np.sum(x*y)/(np.sqrt(np.sum(x*x))*np.sqrt(np.sum(y*y)))

# Read in barriers 
#name = np.genfromtxt("barrier.dat", usecols=1, dtype=str)
outp = np.genfromtxt("barrier.dat", usecols=1, dtype=float)
ligs = np.genfromtxt("barrier.dat", usecols=0, dtype=int)
inp = np.genfromtxt("ligand.desc").transpose()[1:].transpose()

mask = np.full(len(ligs),True)
for i in range(len(ligs)):
	if ( outp[i] == -1 ):
		mask[i] = False

#name = name[mask]
outp = outp[mask]
ligs = ligs[mask]
inp  = inp[mask]

if len(inp) != len(ligs):
	print("barrier.dat doesn't fit ligand.desc\nAborting ....")
	exit()

n_outcomes = 1
n_features = len(inp[0])
n_units =   25
dropout = 0.5

NN = Sequential()

# Add input layer with 25 nodes
NN.add(Dense(n_units, input_shape=(n_features,)))
# NN.add(Dropout(0.8))


# Add 4 hidden layer
NN.add(Dense(45))
NN.add(ReLU())
# NN.add(Dropout(dropout))

NN.add(Dense(10))
NN.add(LayerNormalization())

NN.add(Dense(10))
NN.add(LayerNormalization())
# NN.add(ReLU())
# NN.add(Dropout(dropout))

NN.add(Dense(5))
# NN.add(ReLU())
# NN.add(Dropout(dropout))
# Add output layer and
# Compile with the loss function and optimizer
NN.add(Dense(n_outcomes))
#NN.compile(loss="mean_absolute_error", optimizer='adam', metrics=["mae"])
NN.compile(loss="mean_squared_error", optimizer=opt, metrics=["mse",pearson_r])

print(NN.summary())

ss = StandardScaler()
inp = ss.fit_transform(np.array(inp))
x_t, x_test, y_t, y_test = train_test_split(np.array(inp), np.array(outp), test_size=0.1)
print(x_t.shape)
print(y_t.shape)
# Overwrite splitting for best training (optional)
#x_t = inp
#y_t = outp

# log = NN.fit(x_t, y_t, epochs=2000, batch_size=200, verbose=1, validation_split=0.1)
log = NN.fit(x_t, y_t, epochs=3000, batch_size=200, verbose=1)
#callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-8, mode="auto",restore_best_weights=True, patience=5000000000)]

#print(pred, y_test)
#print("Validation score is ", pred)
plt.plot(log.history["loss"][100:], label="training loss")
# plt.plot(log.history["val_loss"][100:], label="validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
# plt.show()

test = np.transpose(NN.predict(x_test))[0]
test -= 30.10
y_test -= 30.10
# print(" pred.   ref.  diff.")
# for i in range(len(x_test)):
# 	print("{0:6.2f} {1:6.2f} {2:6.2f}".format(test[i],y_test[i],test[i]-y_test[i]))
print("--------------------")
print("        rmsd {0:6.2f}".format(np.sqrt(np.mean(np.square(test-y_test)))))
print("     pearson {0:6.2f}".format(pearson(test,y_test)))
# print(pearsonr(test,y_test))
print("")

exit()

plt.scatter(y_test,test)
plt.xlabel("Reference barrier")
plt.ylabel("Predicted barrier")
plt.show()

s_test = np.full(len(test),0)
s_ytest = np.full(len(y_test),0)

for i in range(len(x_test)):
	if test[i] < 29.10:
		s_test[i] = -1
	if test[i] > 31.10:
		s_test[i] = +1
	if y_test[i] < 29.10:
		s_ytest[i] = -1
	if y_test[i] > 31.10:
		s_ytest[i] = +1
	if s_test[i] == s_ytest[i]:
		buf = "check"
	else:
		buf = "fail: {0:3d} instead of {1:3d}".format(s_test[i],s_ytest[i])
	print("{0:5d} {1:5d} {2:s}".format(s_test[i],s_ytest[i],buf))


