import json
import os
import shutil
import random
from math import floor
from PIL import Image

# 输入你Wildtrack_dataset的路径
base = 'X:/COCO/Multicamera/Wildtrack/Wildtrack_dataset_full/Wildtrack_dataset/'


######################################
# 创建文件夹
def create_folders(base_path):
    folders = ['processed', 'processed_rename', 'processed_rename_rerank', 'camera_split', 'Wildtrack_splited_rerank']
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")

create_folders(base)

# 删除文件夹
def delete_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Folder does not exist: {folder_path}")

##########################################
# read_json.py的部分
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


images = base + 'Image_subsets/'
source = base + 'annotations_positions/'
target = base + 'processed/'

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
                mkdir(target+id)
                personDict[id] = 1
            camera_num = 0
            for camera in person['views']:
                camera_num += 1
                if camera['xmin'] < 0:
                    continue
                pos = (camera['xmin'], camera['ymin'],
                       camera['xmax'], camera['ymax'])

                # 用于wildtrack数据集
                s_path = images+'C'+str(camera_num)+'/'+fil[:8]+'.png'
                t_path = target+id+'/'+fil[4:8]+'_'+str(camera_num)+'.png'

                # 用于MultiviewX数据集
                # s_path = images+'C'+str(camera_num)+'/'+fil[1:5]+'.png'
                # t_path = target+id+'/'+fil[1:5]+'_'+str(camera_num)+'.png'

                save_person(s_path, t_path, pos)
            num += 1
            print(start, 'th file ', num, 'th person finished')
# save_person(images+'0323.png',test+'test2.png',pos[1])

print('finished')

#####################################################################
# rename_png.py的部分

source = base + 'processed'
target = base + 'processed_rename'

indexes = os.listdir(source)
total = 0
for index in indexes:
    ppath = os.path.join(source, index)
    img_paths = os.listdir(ppath)
    name0 = index.zfill(4)
    count = 0
    for img_path in img_paths:
        # print(img_path)
        # sys.exit(0)
        count += 1
        total += 1
        camid = img_path[-5]
        tid = img_path[-10:-6]
        full_tail = name0+'_c'+camid+'_t'+tid+'.png'
        source_path = os.path.join(ppath, img_path)
        target_path = os.path.join(target, full_tail)
        # print(source_path)
        # print(target_path)
        # sys.exit()
        shutil.copy(source_path, target_path)
        print(index, 'person', count, 'th img transfered')
        print('totally', total, 'th img transfered')

delete_folder(source)

################################################
# rerank_id.py的部分

with open('data.json', 'r') as file:
    data = json.load(file)


# processed文件夹的路径
processed_folder = base + 'processed_rename'

# 输出文件夹的路径
output_folder = base + 'processed_rename_rerank'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历processed文件夹中的所有文件
for filename in os.listdir(processed_folder):
    if filename.endswith('.png'):
        parts = filename.split('_')
        if len(parts) != 3:
            print(f"Skipping file with unexpected format: {filename}")
            continue

        original_id = parts[0]
        view_angle = parts[1]
        time_id = parts[2]

        # 去掉时间编号的文件扩展名
        time_id = time_id.split('.')[0]

        # 根据data.json中的映射关系重命名文件
        if original_id in data:
            new_id = data[original_id]
            new_filename = f"{new_id}_{view_angle}_{time_id}.png"
            original_file_path = os.path.join(processed_folder, filename)
            new_file_path = os.path.join(output_folder, new_filename)

            # 复制并重命名文件到新的文件夹
            shutil.copyfile(original_file_path, new_file_path)
            print(f"Copied and renamed {original_file_path} to {new_file_path}")
        else:
            print(f"Original ID {original_id} not found in data.json")

print("Renaming complete.")

delete_folder(processed_folder)

###################################################
# camera_split.py 部分
source = base + 'processed_rename_rerank'
target = base + 'camera_split'

imgs = os.listdir(source)
t = 0
for img in imgs:
    camid = img[6]
    # print(img)
    # print(camid)
    # break
    ori = os.path.join(source, img)
    tar = os.path.join(target, camid, img)

    # 如果目标目录不存在，创建它
    tar_dir = os.path.join(target, camid)
    tar = os.path.join(tar_dir, img)
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    shutil.copy(ori, tar)
    t += 1
    print(t, 'th finished')

delete_folder(source)

#############################################
# Create Wildtrack_splited_rerank
def create_folders(base_path):
    for i in range(1, 8):
        folder_path = os.path.join(base_path, str(i))
        os.makedirs(os.path.join(folder_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'test'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'val'), exist_ok=True)

def split_files(src_folder, dest_folder, train_ratio=0.45, test_ratio=0.45, val_ratio=0.1):
    for i in range(1, 8):
        src_path = os.path.join(src_folder, str(i))
        images = [f for f in os.listdir(src_path) if f.endswith('.png')]
        random.shuffle(images)

        total_images = len(images)
        train_end = floor(total_images * train_ratio)
        test_end = train_end + floor(total_images * test_ratio)

        train_files = images[:train_end]
        test_files = images[train_end:test_end]
        val_files = images[test_end:]

        dest_train_path = os.path.join(dest_folder, str(i), 'train')
        dest_test_path = os.path.join(dest_folder, str(i), 'test')
        dest_val_path = os.path.join(dest_folder, str(i), 'val')

        for file in train_files:
            shutil.copy(os.path.join(src_path, file), os.path.join(dest_train_path, file))

        for file in test_files:
            shutil.copy(os.path.join(src_path, file), os.path.join(dest_test_path, file))

        for file in val_files:
            shutil.copy(os.path.join(src_path, file), os.path.join(dest_val_path, file))

source_folder = base + 'camera_split'
destination_folder = base + 'Wildtrack_splited_rerank'  # 目标文件夹路径

# 创建目标文件夹和子文件夹
create_folders(destination_folder)

# 分割并复制文件
split_files(source_folder, destination_folder)

print("文件已成功复制和分割。")

delete_folder(source_folder)