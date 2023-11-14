from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

# load the FordA dataset

def readucr(filename):
    """Load dataset and corresponding labels. Here the first column contains the label."""
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:,0]
    x = data[:,1:]
    
    return x, y.astype(int)

root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

# visualize samples in 
classes = np.unique(np.concatenate((y_train,y_test), axis=0)) # finds unique elements in array

plt.figure()
for c in classes:

    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class_"+str(c))

plt.legend(loc="best")
plt.show()
plt.close()

# standardize the time series data
# more information: https://link.springer.com/article/10.1007/s10618-016-0483-9 (Bagnall et al., 2016)

# before standardization reshape the time series

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],1))


# To use sparse_categorical_cross_entropy, we will have to count the number of classes
# beforehand

num_classes = len(np.unique(y_train))

# now shuffle the training set, since we will be using the validation_split option
# later when training

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

# now standardize the labels to positive integers: expected labels will then be 0 and 1

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

#----------------------------------------------------------------------------------------
# Build a model with KERAS
#----------------------------------------------------------------------------------------

def make_model(input_shape):

    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs = output_layer)

model = make_model(input_shape = x_train.shape[1:])
#keras.utils.plot_model(model, show_shapes = True)

# train the model

epochs = 10
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr = 0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss",patience=50, verbose=1),   
]

model.compile(
    optimizer="adam",
    loss = "sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split = 0.2,
    verbose = 1
)

# Evaluate Model on test data
model = keras.models.load_model("best_model.h5")
test_loss, test_acc = model.evaluate(x_test,y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

# Plot the training and validation loss of the model

metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_"+metric])
plt.title("model "+metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc = "best")
plt.show()
plt.close()

end = time.time()

print("Execution time:",(end-start)*10**3,"ms")

