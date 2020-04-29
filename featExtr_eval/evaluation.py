import numpy as np
import time
import torch
from featExtr_eval import acc_eval_utils
import csv
import os


def featExtr(imgdir, args, model, transform, target_loader):
    model.eval()
    model.cuda()
    print('start extracting {} images features from: '.format(len(target_loader.dataset)), imgdir)

    feats = [None]*len(target_loader.dataset)
    for i, (img, _) in enumerate(target_loader):
        img = img.cuda()
        if args.get_feats_mode == 'split':
            features = model(img).data  # get the tensor out of the variable
            feaVec = torch.squeeze(features).cpu().numpy()  # convert torch tensor to numpy array

            fd1, fd2 = feaVec.shape[1], feaVec.shape[2]
            f = 0
            for t in range(i * args.batch_size, (i+1)*args.batch_size):
                if t < len(target_loader.dataset):
                    temp = feaVec[f]
                    feats[t] = temp.transpose(2, 1, 0).reshape(fd2 * fd2, fd1)
                    f += 1

        if args.get_feats_mode == 'global':
            features = model(img).data  # get the tensor out of the variable
            feaVec = torch.squeeze(features).cpu().numpy()  # convert torch tensor to numpy array

            f = 0
            for t in range(i * args.batch_size, (i+1)*args.batch_size):
                if t < len(target_loader.dataset):
                    temp = feaVec[f]
                    feats[t] = temp
                    f += 1

    torch.cuda.empty_cache()
    return feats


def featsExtr_eval(args, model, transform, query_loader, target_loader):
    model.eval()
    model.cuda()

    query_n = len(query_loader)
    target_n = len(target_loader.dataset)
    # acc_eval_utils.overlap_table(args, query_loader, target_loader)
    # get the target feature
    feats_t = featExtr(args.target_data, args, model, transform, target_loader)
    vote_res_index = [None] * query_n

    with torch.no_grad():
        for filenum, (img, _) in enumerate(query_loader):
            img = img.cuda()
            start = time.time()

            if args.get_feats_mode == 'split':
                # each row: for each query image patch,
                # each column: record the min patch dis in each target image
                dis_table = np.zeros([len(feats_t[0]), len(feats_t)])
                rot_feats = model(img).data  # get the tensor out of the variable

                rot_feats = torch.squeeze(rot_feats).cpu().numpy()
                fd1, fd2 = rot_feats.shape[0], rot_feats.shape[1]
                query_rot_feats = rot_feats.transpose(2, 1, 0).reshape(fd2 * fd2, fd1)

                for i in range(target_n):
                    min_dis = acc_eval_utils.min_patch_dis(query_rot_feats, feats_t[i])
                    dis_table[:, i] = min_dis

                candidates = dis_table.argmin(axis=1)   # Returns the target indices of the minimum values along row
                # indices of counts are target indices, values are the # . of occurrences
                counts = np.bincount(candidates)    # Count number of occurrences of target indices in array.
                if args.topk == 1:
                    vote_idx = [np.argmax(counts)]    # return the indices of the max values in order
                else:
                    vote_idx = np.argsort(counts)[-args.topk:][::-1]
                vote_res_index[filenum] = vote_idx

            if args.get_feats_mode == 'global':
                dis_table = np.zeros(len(feats_t))
                rot_feats = model(img).data  # get the tensor out of the variable
                rot_feats = torch.squeeze(rot_feats).cpu().numpy()

                for i in range(target_n):
                    min_dis = acc_eval_utils.min_patch_dis(rot_feats, feats_t[i], True)
                    dis_table[i] = min_dis

                vote_idx = dis_table.argmin()  # Returns the target indices of the minimum values along row
                vote_res_index[filenum] = [vote_idx]

        res, acc = acc_eval_utils.compute_acc(vote_res_index, args, filenum)
        # overlap_table = acc_eval_utils.overlap_table(query_loader, target_loader, vote_res_index)
        return acc


def feats_eval_sift(args):
    feats_q, query_n = acc_eval_utils.featExtr_sift(args.query_data, args)
    feats_t, target_n = acc_eval_utils.featExtr_sift(args.target_data, args)

    vote_res_index = [None] * query_n

    for filenum, qimg_feat in enumerate(feats_q):
        start = time.time()

        # each row: for each query image patch,
        # each column: record the min patch dis in each target image
        if qimg_feat is None:
            vote_res_index[filenum] = None
        else:
            dis_table = np.zeros([len(qimg_feat), len(feats_t)])
            for i in range(target_n):
                min_dis = acc_eval_utils.min_patch_dis(qimg_feat, feats_t[i])
                dis_table[:, i] = min_dis

            if filenum % 100 == 0:
                res, acc = acc_eval_utils.compute_acc(vote_res_index, args, filenum)
                print('geolocate: {}/{}'.format(filenum, query_n))
                print('time per geolocalization query: {}'.format(time.time() - start))
            candidates = dis_table.argmin(axis=1)   # Returns the target indices of the minimum values along row
            # indices of counts are target indices, values are the # . of occurrences
            counts = np.bincount(candidates)    # Count number of occurrences of target indices in array.
            if args.topk == 1:
                vote_idx = [np.argmax(counts)]    # return the indices of the max values
            else:
                vote_idx = np.argsort(counts)[-args.topk:][::-1]
            vote_res_index[filenum] = vote_idx
    res, acc = acc_eval_utils.compute_acc(vote_res_index, args, filenum)
    return acc


def get_overlap_hisgram(args, model, transform, query_loader, target_loader, overlap_table):
    model.eval()
    model.cuda()

    query_n = len(query_loader)
    target_n = len(target_loader.dataset)

    target_info = os.path.join(args.target_data, 'target_info.csv')
    tname_csv, tleft_csv, tright_csv, tupper_csv, tlower_csv, len_tinfo = acc_eval_utils.load_info(target_info)

    feats_t = featExtr(args.target_data, args, model, transform, target_loader)
    vote_res_index = [None] * query_n
    with open(overlap_table, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data_list = list(reader)

    # with open('/home/yuxin/PycharmProjects/wacv2020_2nd/failure cases799.csv', newline='') as csvfile:
    #     reader = csv.reader(csvfile)
    #     fail_list = list(reader)

    overlap_dict = {'0': {'success': 0, 'fail': 0}, '10': {'success': 0, 'fail': 0}, '20': {'success': 0, 'fail': 0},
                    '30': {'success': 0, 'fail': 0}, '40': {'success': 0, 'fail': 0}, '50': {'success': 0, 'fail': 0},
                    '60': {'success': 0, 'fail': 0}, '70': {'success': 0, 'fail': 0}, '80': {'success': 0, 'fail': 0},
                    '90': {'success': 0, 'fail': 0}, '100': {'success': 0, 'fail': 0}}
    s = 0
    f = 0
    with torch.no_grad():
        for filenum, (img, _) in enumerate(query_loader):
            img = img.cuda()
            dis_table = np.zeros([len(feats_t[0]), len(feats_t)])

            rot_feats = model(img).data  # get the tensor out of the variable
            rot_feats = torch.squeeze(rot_feats).cpu().numpy()
            fd1, fd2 = rot_feats.shape[0], rot_feats.shape[1]
            query_rot_feats = rot_feats.transpose(2, 1, 0).reshape(fd2 * fd2, fd1)

            for i in range(target_n):  # i is the index of target, start from 0, not 1
                min_dis = acc_eval_utils.min_patch_dis(query_rot_feats, feats_t[i])
                dis_table[:, i] = min_dis

            candidates = dis_table.argmin(axis=1)  # Returns the target indices of the minimum values along row
            # indices of counts are target indices, values are the # . of occurrences
            counts = np.bincount(candidates)  # Count number of occurrences of target indices in array.

            target_num = int(data_list[filenum + 1][4])
            vote_idx = np.argsort(counts)[-target_num:][::-1]   # t_name = tname_csv[tile_idx + 1, 0]

            vote_res_index[filenum] = vote_idx

            s_bool = False
            for truth_idx in range(5, len(data_list[filenum + 1][5:]), 3):      # t_name = tname_csv[t_index + 1, 0]
                area = float(data_list[filenum + 1][truth_idx + 2])
                lap_perc = int(round(area / (256 ** 2) * 100, -1))

                temp = int(data_list[filenum + 1][truth_idx])
                if vote_idx[0] == temp:
                    s_bool = True
                if int(data_list[filenum + 1][truth_idx]) in vote_idx:
                    overlap_dict[str(lap_perc)]['success'] += 1
                else:
                    overlap_dict[str(lap_perc)]['fail'] += 1
            if s_bool:
                s = s + 1
            else:
                f = f + 1
            # s_old = fail_list[filenum + 1][3] == '_'

            # if s_bool != s_old:
            #     print('filenum is: {}\tvote_idx is {}'.format(filenum, vote_idx))

    print('s: ' + str(s))
    print(overlap_dict)
    print('over')
