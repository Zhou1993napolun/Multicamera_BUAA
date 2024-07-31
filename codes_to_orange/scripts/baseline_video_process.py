import numpy as np
import os

root = '../orange_demo/video/0134_c6_t0860to0945/'

# from collections import defaultdict
frame_order = {}
paths = []
pids = []
cnt = 0


with open(os.path.join(root,'img_path.txt'), 'r') as f:
    for path in f.readlines():
        p = path.strip()
        # print(p)
        paths.append(p)
        # print(p.strip('/'))
        # print(p.strip('/')[-1][:4])
        pid = int(p.split('/')[-1][:4])
        pids.append(pid)
        # 第一张为query
        if not cnt:
            cnt += 1
            continue
        cnt += 1
        time = p.split('/')[-1][-8:-4]
        if time not in frame_order:
            frame_order[time] = 1
        else:
            frame_order[time] += 1

num_query = 1
num_gallery = len(paths) - num_query
print(frame_order)
print(num_gallery)


dist = np.load(os.path.join(root, 'dist.npy'))
print(dist.shape)

dists = []
start = 0
end = 0
for _, v in frame_order.items():
    end += v
    cur_d = dist[:, start:end]
    # print(start, end, cur_d.shape)
    dists.append(cur_d)
    start = end

print(len(dists[0][0]))

def sort_rows_and_get_indices(array):
    # 获取每一行排序后的索引
    # sorted_indices = np.argsort(array, axis=1)[:, ::-1]
    sorted_indices = np.argsort(array, axis=1)


    # 对每一行进行排序
    sorted_rows = np.take_along_axis(array, sorted_indices, axis=1)

    return sorted_rows, sorted_indices

# 举例一个二维数组
example_array = np.array([[3, 1, 4, 1],
                          [5, 9, 2, 6],
                          [5, 3, 5, 8]])


sorted_dists = []
sorted_indices = []

for d in dists:
    sorted_dist, sorted_indice = sort_rows_and_get_indices(d)
    sorted_dists.append(sorted_dist)
    sorted_indices.append(sorted_indice)

for s in sorted_dists:
    if s.size == 0:
        print("s is empty")
    print(s[0][0])


from tqdm import tqdm
def get_top_k_similar_pids(array, g_start=0, k=0):
    top_k_matches = []
    for q_id, g_ids in tqdm(enumerate(sorted_indice)):
        q_pid = pids[q_id]
        # q_camid = camids[q_id]
        l = 0
        for g_id in g_ids:
            if l >= k:
                break
            # g_camid = camids[num_query + g_id]
            # if q_camid == g_camid:
            #     continue
            g_pid = pids[num_query + g_start + g_id]
            tmp_list=paths[num_query + g_start + g_id], g_pid==q_pid
            l += 1
        top_k_matches.append(tmp_list)
    return top_k_matches

top_5_matches_list = []
g_start = 0
for sorted_indice in sorted_indices:
    top_5_matches = get_top_k_similar_pids(sorted_indice, g_start, 1)
    g_start += sorted_indice.shape[1]
    top_5_matches_list.append(top_5_matches)


trues = 0
cnt = 0
for id, item in enumerate(top_5_matches_list):
    # print(item)
    name, preci = item[0]
    if preci:
        trues += 1
    else:
        print(name, id)
    cnt += 1

id = 0
s = 0
for k, v in frame_order.items():
    s += v
    print(id,k)
    id += 1
print(s)


with open(os.path.join(root, 'results.txt'), 'w') as f:
    for id, item in enumerate(top_5_matches_list):
        # print(item)
        name, preci = item[0]
        f.write(name)
        f.write('\t')
        f.write(str(preci))
        f.write('\t')
        f.write(str(sorted_dists[id][0][0]))
        f.write('\n')














