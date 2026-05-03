# config.py

# Signal parameters
FS_LOW = 2048       # input sample rate
FS_HIGH = 8192      # target sample rate (4x)
DURATION = 0.05     # signal duration (seconds)

# Training parameters
NUM_TRAIN_SAMPLES = 5000
BATCH_SIZE = 64
EPOCHS = 600
LEARNING_RATE = 0.0001
SEED = 42

# Paths
MODEL_PATH = "residual_model.pth"
SCALER_PATH = "scaler_params.npz"