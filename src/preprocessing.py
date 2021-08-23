import os

from PIL import Image, ImageFilter



def bluring_and_resize():
    dest = 'validation'
    repo = 'photos'
    repo2 = 'photosPrepro'
    for i in os.listdir(f'../ressources/{repo}/{dest}'):
        with Image.open(f'../ressources/{repo}/{dest}/' + i) as image:
            image = image.convert('RGB')
            image = image.filter(ImageFilter.GaussianBlur(radius=2))
            image = image.resize((240, 240))

            image.save(f'../ressources/{repo2}/{dest}/' + i)


def change_label(oldDirectory, newDirectory, oldLabel, newLabel):
    for file in os.listdir(oldDirectory):
        if int(file.split('_')[0]) == oldLabel:
            old_file = os.path.join(newDirectory, file)
            new1 = file.split('_')[1]
            new = f'{newLabel}_{new1}'
            new_file = os.path.join(newDirectory, new)
            os.rename(old_file, new_file)
