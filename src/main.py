# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random

from cnn import train_and_save_model


directory = f'../ressources/photos/training'

if __name__ == '__main__':
   train_and_save_model()
#relable(directory, directory, 5, 4)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
