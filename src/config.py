import os

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

gpus = [0]
batch_size = 32
fce = True
classes_per_set = 4
samples_per_class = 5
channels = 128
# Training setup
total_epochs = 500
total_train_batches = 1000
total_val_batches = 100
total_test_batches = 250
n_test_samples = 4

model_path = ROOT_DIR + '/resource/model/'
