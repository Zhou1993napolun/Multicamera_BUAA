import json
import os

from PIL import Image


def save_person(source, target, position):
    img = Image.open(source)
    out = img.crop(position)
    out.save(target)


def mkdir(path):
    path = path.strip()
    path = path.rstrip('\\')
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print('succeed creating dir')
    else:
        print('dir already existed')


images = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/Image_subsets/'
source = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/annotations_positions/'
target = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/processed/'
start = 0
personDict = dict()
files = os.listdir(source)
for fil in files:
    start += 1
    name = os.path.join(source, fil)
    print(name)
    with open(name, 'r') as f:
        j_data = json.load(f)
        num = 0
        for person in j_data:
            id = str(person['personID'])
            if id not in personDict:
                mkdir(target + id)
                personDict[id] = 1
            camera_num = 0
            for camera in person['views']:
                camera_num += 1
                if camera['xmin'] < 0:
                    continue
                pos = (camera['xmin'], camera['ymin'],
                       camera['xmax'], camera['ymax'])
                s_path = images + 'C' + str(camera_num) + '/' + fil[:8] + '.png'
                t_path = target + id + '/' + fil[4:8] + '_' + str(camera_num) + '.png'
                save_person(s_path, t_path, pos)
            num += 1
            print(start, 'th file ', num, 'th person finished')
# save_person(images+'0323.png',test+'test2.png',pos[1])

print('finished')
