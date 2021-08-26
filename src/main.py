# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import random

from cnn import train_and_save_model
import tensorflow as tf

from preprocessing import change_label

directory = f'../ressources/Food-5k/training'
directory2 = f'../ressources/photos/training'
USE_GPU = 1



def setup_gpus(selected_gpu_ids: list, use_cuda_visible_devices: bool = False, reserved_memory: int = None,
               verbosity: int = 0):
   # don't use this command: print("Are GPUs available?", tf.test.is_gpu_available())
   # it seems to put the gpu handlers in a bad state
   print("Trying to setup gpu board(s): {}".format(selected_gpu_ids))
   # not mandatory to set CUDA_VISIBLE_DEVICES but it can be
   # useful as it completely prevents TensorFlow from using other GPUs.
   if use_cuda_visible_devices:
      os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(idx) for idx in selected_gpu_ids)
   gpus = tf.config.experimental.list_physical_devices('GPU')
   selected_gpus = list()
   if gpus:
      print('Available GPUS on this server: {}'.format(gpus))
      if use_cuda_visible_devices:
         selected_gpus = gpus
      else:
         for idx in selected_gpu_ids:
            selected_gpus.append(gpus[idx])
      print('Selected GPUS: {}'.format(selected_gpus))
      try:
         # Limit memory reserved by TensorFlow on selected gpus to only the needed amount or
         # reserved_memory if specified
         if reserved_memory is None:
            for gpu in selected_gpus:
               tf.config.experimental.set_memory_growth(gpu, True)
         else:
            for gpu in selected_gpus:
               tf.config.experimental.set_virtual_device_configuration(gpu,
                                                                       [
                                                                          tf.config.experimental.VirtualDeviceConfiguration(
                                                                             memory_limit=reserved_memory)])

         # Restrict TensorFlow to only use seleced gpus
         # Not necessary if use_cuda_visible_devices is True as selected_gpus == gpus
         if not use_cuda_visible_devices:
            #if verbosity > 0:
               #logging.info('Restricting TF to the selected GPUS')
            tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')
         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
         print("{} Physical GPUs, {} Logical GPUs".format(len(gpus), len(logical_gpus)))
      except RuntimeError as e:
         # Visible devices must be set before GPUs have been initialized
         print(e)
      print("Gpus are now set to be used by TensorFlow")
   else:
      print("Sorry no gpu available on this host, remove -g flag to use cpu")
      exit(1)


def test():
   list = [16, 32, 64, 128, 256, 512, 1024]
   listAcc = [0, 0, 0, 0, 0, 0, 0]
   nb = 10
   for i in range(7):
      for j in range(nb):
         listAcc[i] += list[i]
   print(listAcc)

if __name__ == '__main__':

   if USE_GPU ==1 :
      setup_gpus([0], verbosity=1)
   train_and_save_model()


   #change_label(directory, directory2, 1, 4)

# See PyCharm # at https://www.jetbrains.com/help/pycharm/
#source venv/bin/activate
