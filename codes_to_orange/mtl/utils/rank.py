import numpy as np
from tqdm import tqdm


def rank_by_distance(ordered_dists, matches, max_rank=50):
    # 目前已有：按照dist排序得到的从小到大的distmat和对应indice
    # +在该indice下正确匹配的位置。可以将列表中前max_rank个dist进行比较
    # 保留最大的那个对应的g_targets中对应indice下的id，作为predict。再和gt
    # 比较记录该frame下match正误
    distmat = [i.T[:max_rank] for i in ordered_dists]
    matchmat = [i.T[:max_rank] for i in matches]
    distmat = np.concatenate(distmat, axis=1).T  # (6,20)
    matchmat = np.concatenate(matchmat, axis=1).T  # (6,20)
    desire_shape = distmat.shape
    order = np.argmin(distmat, axis=0)  # 得到每一个rank下最小的dist对应的cam编号
    cur_matches = matchmat[order, np.arange(matchmat.shape[1])]
    cur_matches = np.tile(cur_matches, desire_shape[0]).reshape(desire_shape)
    return cur_matches  # (20,)× # (6,20)


def rank_by_vote(predicts, matches, max_rank=50):
    predmat = [i.T[:max_rank] for i in predicts]
    matchmat = [i.T[:max_rank] for i in matches]
    predmat = np.concatenate(predmat, axis=1).T  # (6,20)
    matchmat = np.concatenate(matchmat, axis=1).T  # (6,20)
    desire_shape = predmat.shape
    most_frequent = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=0, arr=predmat)
    # 获取每列中出现最多的数字在列中第一次出现的索引
    first_occurrences = []
    for idx, col in enumerate(predmat.T):  # 遍历每列
        max_idx = np.where(col == most_frequent[idx])[0][0]
        first_occurrences.append(max_idx)
    order = np.array(first_occurrences)
    cur_matches = matchmat[order, np.arange(matchmat.shape[1])]
    cur_matches = np.tile(cur_matches, desire_shape[0]).reshape(desire_shape)
    return cur_matches


def rank(matches, max_rank=50):
    matchmat = [i.T[:max_rank] for i in matches]
    matchmat = np.concatenate(matchmat, axis=1).T  # (6,20)
    return matchmat


def evaluate_wild(dist, q_targets, g_targets, q_cameras, max_rank=10):
    tmp_q_targets = []
    for batch in q_targets:
        tmp_batch = []
        for cam in batch:
            tmp_batch.append(cam.cpu().numpy())
        tmp_q_targets.append(tmp_batch)
    q_targets = tmp_q_targets
    tmp_g_targets = {}
    for key, targets in g_targets.items():
        tmp_list = []
        for target in targets:
            tmp_list.append(target.cpu().numpy())
        tmp_g_targets[key] = np.array(tmp_list)
    g_targets = tmp_g_targets

    total_matches_by_distance = []
    total_matches_by_vote = []
    total_matches_original = []
    for frame_idx, frame_batch in enumerate(tqdm(dist)):
        predicts = []
        ordered_dists = []
        matches = []
        for cam_idx, cam_dist in enumerate(frame_batch):
            num_q, num_g = cam_dist.shape

            # 目前不需要判断
            # if num_g < max_rank:
            #     max_rank = num_g
            #     print(
            #         'Note: number of gallery samples is quite small, got {}'.format(num_g))

            indice = np.argsort(cam_dist, axis=1)
            q_pid = q_targets[frame_idx][cam_idx]
            camid = q_cameras[frame_idx][cam_idx]

            ordered_dist = cam_dist[0][indice]

            predict = g_targets[camid][indice]
            match = (predict == q_pid).astype(np.int32)

            predicts.append(predict)
            ordered_dists.append(ordered_dist)
            matches.append(match)

        current_matches_by_distance = rank_by_distance(
            ordered_dists, matches, max_rank)
        total_matches_by_distance.append(current_matches_by_distance)

        current_matches_by_vote = rank_by_vote(predicts, matches, max_rank)
        total_matches_by_vote.append(current_matches_by_vote)

        current_matches_original = rank(matches, max_rank)
        total_matches_original.append(current_matches_original)
    # list,len():850
    return total_matches_by_distance, total_matches_by_vote, total_matches_original


# 后续在这改
def evaluate_wild_mmoe_all(dist, q_targets, g_targets, max_rank=10):
    q_targets = [item.repeat(7) for item in q_targets]
    tmp_q_targets = []
    for batch in q_targets:
        tmp_batch = []
        for cam in batch:
            tmp_batch.append(cam.cpu().numpy())
        tmp_q_targets.append(tmp_batch)
    q_targets = tmp_q_targets
    # tmp_q_targets = [arr.cpu().numpy() for arr in q_targets]
    # tmp_q_targets = np.concatenate([np.expand_dims(arr, 1) for arr in tmp_q_targets], axis=0)
    # nums = tmp_q_targets.shape[0]
    # tmp_q_targets = np.broadcast_to(tmp_q_targets, (nums, 7))
    # tmp_q_targets = np.split(tmp_q_targets, nums, axis=0)
    # tmp = []
    # for ele in tmp_q_targets:
    #     tmp.append([np.array[item] for item in ele])
    # q_targets = tmp

    # q_targets = tmp_q_targets

    tmp_g_targets = {}
    for key, targets in g_targets.items():
        tmp_list = []
        for target in targets:
            tmp_list.append(target.cpu().numpy())
        tmp_g_targets[key] = np.array(tmp_list)
    g_targets = tmp_g_targets

    tmp_dist = []
    nums_query = len(dist[0])
    for i in range(nums_query):
        query_list = []
        for j in range(len(dist)):
            nums_gallery = dist[j][i].shape[0]
            reshaped_array = dist[j][i].reshape(1, nums_gallery)
            query_list.append(reshaped_array)
        tmp_dist.append(query_list)
    dist = tmp_dist

    total_matches_by_distance = []
    total_matches_by_vote = []
    total_matches_original = []
    for frame_idx, frame_batch in enumerate(tqdm(dist)):
        predicts = []
        ordered_dists = []
        matches = []
        for cam_idx, cam_dist in enumerate(frame_batch):
            num_q, num_g = cam_dist.shape

            indice = np.argsort(cam_dist, axis=1)
            q_pid = q_targets[frame_idx][cam_idx]
            camid = cam_idx + 1

            ordered_dist = cam_dist[0][indice]

            predict = g_targets[camid][indice]
            match = (predict == q_pid).astype(np.int32)

            predicts.append(predict)
            ordered_dists.append(ordered_dist)
            matches.append(match)

        current_matches_by_distance = rank_by_distance(
            ordered_dists, matches, max_rank)
        total_matches_by_distance.append(current_matches_by_distance)

        current_matches_by_vote = rank_by_vote(predicts, matches, max_rank)
        total_matches_by_vote.append(current_matches_by_vote)

        current_matches_original = rank(matches, max_rank)
        total_matches_original.append(current_matches_original)
    # list,len():850
    return total_matches_by_distance, total_matches_by_vote, total_matches_original


def evaluate_wild_mmoe(distmat, q_pids, g_pids, max_rank=20):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    g_pids = [item.to('cpu').item() for item in g_pids]
    g_pids = np.array(g_pids)
    q_pids = [item.to('cpu').item() for item in q_pids]
    q_pids = np.array(q_pids)

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in tqdm(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid + 1000000)  # 偷懒，直接让这个不remove
        keep = np.invert(remove)

        # compute cmc curve
        matches = (g_pids[order] == q_pid).astype(np.int32)
        # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[keep]
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    mINP = np.mean(all_INP) * 100
    mAP = np.mean(all_AP) * 100
    ranks = {}
    for r in [1, 5, 10]:
        ranks['Rank-{}'.format(r)] = all_cmc[r - 1] * 100

    return mAP, mINP, ranks


def calculate_score(matches, q_cameras):
    assert len(matches) == len(q_cameras), 'invalid input'
    sub_matches = {}
    n = len(matches)
    for idx in range(n):
        for camid, cam in enumerate(q_cameras[idx]):
            if cam not in sub_matches:
                sub_matches[cam] = [matches[idx][camid]]
            else:
                sub_matches[cam].append(matches[idx][camid])
    scores = {}
    for k, l in sub_matches.items():
        scores[k] = {}
        res = scores[k]
        all_cmc = []
        all_AP = []
        all_INP = []
        num_valid_q = 0.
        for raw_cmc in l:
            if not np.any(raw_cmc):
                continue
            cmc = raw_cmc.cumsum()

            pos_idx = np.where(raw_cmc == 1)
            max_pos_idx = np.max(pos_idx)
            inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
            all_INP.append(inp)

            cmc[cmc > 1] = 1

            all_cmc.append(cmc)
            num_valid_q += 1.

            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q

        print(str(k), 'th camera')
        # print('mAP:', np.mean(all_AP) * 100)
        print('minp: {:.2f}'.format(np.mean(all_INP) * 100))
        print('mAP: {:.2f}'.format(np.mean(all_AP) * 100))
        res['mAP'] = np.mean(all_AP) * 100
        for r in [1, 5, 10]:
            res['Rank-{}'.format(r)] = all_cmc[r - 1] * 100
            # print('Rank-', str(r), ':', all_cmc[r - 1] * 100)
            print('Rank-', str(r), ': {:.2f}'.format(all_cmc[r - 1] * 100))


def calculate_score_by_group(matches):
    new_matches = []
    n = len(matches)
    for idx in range(n):
        new_matches.append(matches[idx][0])

    res = {}
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    for raw_cmc in new_matches:
        if not np.any(raw_cmc):
            continue
        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc)
        num_valid_q += 1.

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    # print('mAP:', np.mean(all_AP) * 100)
    print('minp: {:.2f}'.format(np.mean(all_INP) * 100))
    print('mAP: {:.2f}'.format(np.mean(all_AP) * 100))
    res['mAP'] = np.mean(all_AP) * 100
    for r in [1, 5, 10]:
        res['Rank-{}'.format(r)] = all_cmc[r - 1] * 100
        # print('Rank-', str(r), ':', all_cmc[r - 1] * 100)
        print('Rank-', str(r), ': {:.2f}'.format(all_cmc[r - 1] * 100))


def calculate_score_mmoe_with_gallery_all(matches, num_cams=7):
    # assert len(matches) == len(q_cameras), 'invalid input'
    sub_matches = {}
    n = len(matches)
    cameras = [i + 1 for i in range(num_cams)]
    for idx in range(n):
        for camid, cam in enumerate(cameras):
            if cam not in sub_matches:
                sub_matches[cam] = [matches[idx][camid]]
            else:
                sub_matches[cam].append(matches[idx][camid])
    scores = {}
    for k, l in sub_matches.items():
        scores[k] = {}
        res = scores[k]
        all_cmc = []
        all_AP = []
        num_valid_q = 0.
        for raw_cmc in l:
            if not np.any(raw_cmc):
                continue
            cmc = raw_cmc.cumsum()

            cmc[cmc > 1] = 1

            all_cmc.append(cmc)
            num_valid_q += 1.

            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q

        print(str(k), 'th camera')
        # print('mAP:', np.mean(all_AP) * 100)
        print('mAP: {:.2f}'.format(np.mean(all_AP) * 100))
        res['mAP'] = np.mean(all_AP) * 100
        for r in [1, 5, 10]:
            res['Rank-{}'.format(r)] = all_cmc[r - 1] * 100
            # print('Rank-', str(r), ':', all_cmc[r - 1] * 100)
            print('Rank-', str(r), ': {:.2f}'.format(all_cmc[r - 1] * 100))
