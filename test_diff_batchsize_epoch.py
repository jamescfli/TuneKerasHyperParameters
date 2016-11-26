# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV    # sklearn is a wrapper for scikit-learn 0.18.1
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("./datasets/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]      # limited by GPU memory
epochs = [10, 50, 100]                      # better go with early-stop settings
param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
# n_jobs: number of jobs to run in parallel
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)		# use parallelism (n_jobs=-1)
# INFO (theano.gof.compilelock): Waiting for existing lock by process '65624' (I am process '65627')
# INFO (theano.gof.compilelock): Waiting for existing lock by process '65624' (I am process '65629')
# switch back to n_jobs = 1
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))