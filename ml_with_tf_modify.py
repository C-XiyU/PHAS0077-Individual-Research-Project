import numpy as np 
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers,activations
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,LSTM

from sklearn.cluster import KMeans
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
n_units =   20
dropout = 0.2
def get_model():
    model = tf.keras.Sequential([
        LSTM(80, return_sequences=True),
        Dropout(0.8),
		LSTM(120,return_sequences=True),
		Dropout(0.2),
        LSTM(120),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='mse')  # 损失函数用均方误差
    # 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
    return model
def get_model_2():
    model = tf.keras.Sequential([
        SimpleRNN(n_units, return_sequences=True),
		Dropout(0.8),
        SimpleRNN(190, return_sequences=True),
        Dropout(0.1),
		SimpleRNN(120, return_sequences=True),
		Dropout(0.1),
		SimpleRNN(150, return_sequences=True),
		Dropout(0.1),
		SimpleRNN(180),
		Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='mse')  # 损失函数用均方误差
    # 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
    return model
def get_model_3():
	NN = Sequential()
	NN.add(Dense(80, input_shape=(n_features,), activation="linear"))
	NN.add(Dropout(0.8))
	NN.add(layers.BatchNormalization())
	NN.add(Dense(80, activation="relu"))
	NN.add(Dropout(0.2))
	for i in range(28):
		NN.add(Dense(120, activation="linear"))
		NN.add(Dropout(0.2))
	NN.add(Dense(120, activation="linear"))
	NN.add(Dropout(0.2))
	# Add output layer and
	# Compile with the loss function and optimizer
	NN.add(Dense(1, activation='linear'))
	NN.compile(loss="mean_squared_error", optimizer='adam', metrics=["mae"])
	return NN
# NN = Sequential()
#
# # Add Dense layer with 100 units
# NN.add(Dense(n_units, input_shape=(n_features,), activation="linear"))
# NN.add(Dropout(0.7))
# NN.add(layers.BatchNormalization())
# for i in range(16):
# 	NN.add(Dense(n_units, activation="linear"))
# 	NN.add(Dropout(0.2))
# # Add output layer and
# # Compile with the loss function and optimizer
# NN.add(Dense(n_outcomes, activation='linear'))
# NN.compile(loss="mean_absolute_error", optimizer='adam', metrics=["mae"])
#NN.compile(loss="mean_squared_error", optimizer='adam', metrics=["mse"])
NN=get_model()
labels=[]
# for dat in outp:
# 	if dat < 29.10:
# 		labels.append(0)
# 	if dat > 31.10:
# 		labels.append(1)
# 	if dat < 29.10:
# 		labels.append(2)
# 	if dat > 31.10:
# 		labels.append(3)
kmeans = KMeans(n_clusters=3, random_state=0)
labels=kmeans.fit(outp.reshape(-1, 1))
x_t, x_test, y_t, y_test = train_test_split(np.array(inp).reshape(-1,4,3), np.array(labels.labels_), test_size=0.1,random_state=32)

# plt.scatter(np.arange(0,labels.labels_.shape[0]),labels.labels_)
# Overwrite splitting for best training (optional)
#x_t = inp
#y_t = outp

log = NN.fit(x_t, y_t, epochs=20, batch_size=200, verbose=0, validation_split=0.1)
print(NN.summary())
#callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-8, mode="auto",restore_best_weights=True, patience=5000000000)]

#print(pred, y_test)
#print("Validation score is ", pred)
plt.plot(log.history["loss"][100:], label="training loss")
plt.plot(log.history["val_loss"][100:], label="validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
#plt.show()

test = np.transpose(NN.predict(x_test))[0]
# test -= 30.10
# y_test -= 30.10
# test -= 1
# y_test -= 1
print(" pred.   ref.  diff.")
for i in range(len(x_test)):
	print("{0:6.2f} {1:6.2f} {2:6.2f}".format(test[i],y_test[i],test[i]-y_test[i]))
print("--------------------")
print("        rmsd {0:6.2f}".format(np.sqrt(np.mean(np.square(test-y_test)))))
print("     pearson {0:6.2f}".format(pearson(test,y_test)))
print("")

x=np.arange(0,y_test.shape[0])

plt.plot(x,y_test,label='true')
plt.plot(x,test,label='predict')
plt.xlabel("Reference barrier")
plt.ylabel("Predicted barrier")
plt.legend()
plt.show()
exit()
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


