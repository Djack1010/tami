# Cati

CATI (Converter Apk To Image) is a project implemented by Christian Peluso, student of Information Technology in Pesche (IS).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Create a virtual environment with Python3, than activate the script and install the dependencies written in the requirements file.

Than install apktool and add the PATH to the environment variables to be able to decompile the apk.

Move your apk file in the folder "sample", once there you can ride apk_decompiler.py in the "py" folder to extract the smali files, 

finally you can run the main that will convert the smalis in OPCode and than in PNG.

### Dependencies

The project needs Python3 to be run, and it has been tested only in Windows 10.

You also need to set the variable 'main_path' in utils/config.py to the full path to the repository folder on your local machine.

#### External tools required for vectorization:
GIST DESCRIPTOR


#### Tips for GPU usages

For problems with the convolutional algorithm you should try the following code:

```python
#TF >= 2.x
 gpus = tf.config.experimental.list_physical_devices('GPU')
 if gpus:
     try:
         # Currently, memory growth needs to be the same across GPUs
         for gpu in gpus:
             tf.config.experimental.set_memory_growth(gpu, True)
         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
     except RuntimeError as e:
         # Memory growth must be set before GPUs have been initialized
         print(e)
#Or
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
```