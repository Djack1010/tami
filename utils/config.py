import time

# MAIN Absolute Path, set at runtime
main_path = None
# Timestamp Execution
timeExec = "{}".format(time.strftime("%d%m_%H%M%S"))

# GLOBAL variables
AUTOTUNE, CHANNELS, IMG_DIM, VECTOR_DIM, CLASS_NAMES, BATCH_SIZE, DATA_REQ, LEARNING_RATE = \
    None, None, None, None, None, None, None, None
