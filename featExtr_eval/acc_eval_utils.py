import cv2
import csv
import os
from PIL import Image
import numpy as np
import time
import torch
import shutil
import geolocate_utils
from utils import eucDis
import shapely.geometry
import shapely.affinity


TILE_SIZE = 256


class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle  # degree

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())


def load_info(testinfo):
    # load the query image info
    name_csv = np.array(['query_patch_name'], dtype=object)
    left_csv = np.array(['left'], )
    right_csv = np.array(['right'], )
    upper_csv = np.array(['upper'], )
    lower_csv = np.array(['lower'], )

    with open(testinfo, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['query_patch_name']
            left = row['left']
            right = row['right']
            upper = row['upper']
            lower = row['lower']

            name_csv = np.vstack([name_csv, name])
            left_csv = np.vstack([left_csv, left])
            right_csv = np.vstack([right_csv, right])
            upper_csv = np.vstack([upper_csv, upper])
            lower_csv = np.vstack([lower_csv, lower])
    len_info = len(name_csv)-1
    return name_csv, left_csv, right_csv, upper_csv, lower_csv, len_info


def load_info_center(testinfo):
    # load the query image info
    name_csv = np.array(['query_patch_name'], dtype=object)
    center_row_csv = np.array(['center_row'], )
    center_col_csv = np.array(['center_col'], )
    angle_csv = np.array(['angle'],)

    with open(testinfo, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['query_patch_name']
            c_row = row['center_row']
            c_col = row['center_col']
            angle = row['angle']

            name_csv = np.vstack([name_csv, name])
            center_row_csv = np.vstack([center_row_csv, c_row])
            center_col_csv = np.vstack([center_col_csv, c_col])
            angle_csv = np.vstack([angle_csv, angle])

    len_info = len(name_csv)-1
    return name_csv, center_row_csv, center_col_csv, len_info, angle_csv


def load_target_num(testinfo):
    # load the query image info
    name_csv = np.array(['query_patch_name'], dtype=object)
    center_row_csv = np.array(['center_row'], )
    center_col_csv = np.array(['center_col'], )
    angle_csv = np.array(['angle'],)
    target_num_csv = np.array(['target_num'], )

    with open(testinfo, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['query_patch_name']
            c_row = row['center_row']
            c_col = row['center_col']
            angle = row['angle']
            target_num = row['target_num']

            name_csv = np.vstack([name_csv, name])
            center_row_csv = np.vstack([center_row_csv, c_row])
            center_col_csv = np.vstack([center_col_csv, c_col])
            angle_csv = np.vstack([angle_csv, angle])
            target_num_csv = np.vstack([target_num_csv, target_num])

    len_info = len(name_csv)-1
    return name_csv, center_row_csv, center_col_csv, len_info, angle_csv, target_num


def is_overlap_center(tile_size, q_cen_row, q_cen_col, t_upper, t_lower, t_left, t_right):
    diag = tile_size * np.sqrt(2)

    t_cen_row = (t_upper + t_lower) / 2
    t_cen_col = (t_left + t_right) / 2

    t = np.array([t_cen_row, t_cen_col])
    q = np.array([q_cen_row, q_cen_col])

    dis2 = np.linalg.norm(t-q)

    success = 1 if dis2 <= diag else 0

    return success


def is_overlap(tile_size, q_cen_row, q_cen_col, t_upper, t_lower, t_left, t_right, angle):
    t_cen_row = (t_upper + t_lower) / 2
    t_cen_col = (t_left + t_right) / 2

    query = RotatedRect(cx=q_cen_col, cy=q_cen_row, w=tile_size, h=tile_size, angle=angle)
    target = RotatedRect(cx=t_cen_col, cy=t_cen_row, w=tile_size, h=tile_size, angle=0)

    overlap_area = query.intersection(target).area
    success = 1 if overlap_area > 0. else 0
    # if overlap_area > 0:
    #     success = 1
    # else:
    #     success = 0
    return success, overlap_area


def extr_sift(infile):
    img = cv2.imread(infile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptor = sift.detectAndCompute(gray, None)
    return kp, descriptor


def featExtr_sift(imgdir, args):
    # load all the tiles in the folder
    # filelist, n = geolocate_utils.load_img(imgdir)

    filelist = glob.glob(os.path.join(imgdir, '*.tif'))
    filelist.sort()
    n = len(filelist)

    if n == 0:
        raise ValueError('wrong imgdir for extracting SIFT feature')
    print('start extracting {} images features from: '.format(n), imgdir)

    feats = [None]*n
    for filenum, infile in enumerate(filelist):
        if args.get_feats_mode == 'sift':
            kp, descriptor = extr_sift(infile)
            if descriptor is not None:
                descriptor = descriptor

            feats[filenum] = descriptor
    return feats, n


def min_patch_dis(qimg_feat, timg_feat, g=False):
    """
    :param query_img_feats: features from one query image, stored in numpy array as shape (fd2*fd2, fd1)
    :param target_feats: features from one target image, shape (fd2*fd2, fd1)
    :return: for each split feature of query, the min dis of split feature of target
    """
    if timg_feat is None:
        min_dis = [999999] * len(qimg_feat)
    else:
        if g:
            min_dis = eucDis.global_l2(qimg_feat, timg_feat)
        else:
            dis_temp = eucDis.pairwise_distances(qimg_feat, timg_feat)
            # for each patch, smallest target patch dis in target image
            min_dis = dis_temp.min(axis=1)
    return min_dis


def compute_acc(vote_res_index, args, filenum, tile_size=256):
    # 1. load image tiles location information
    query_info = os.path.join(args.query_data, 'query_info.csv')
    target_info = os.path.join(args.target_data, 'target_info.csv')
    qname_csv, q_center_row_csv, q_center_col_csv, len_qinfo, angle_csv = load_info_center(query_info)
    tname_csv, tleft_csv, tright_csv, tupper_csv, tlower_csv, len_tinfo = load_info(target_info)

    # 2. loop all the query image tile to determine if it is success
    res = np.empty([len(vote_res_index), 11], dtype=object)
    # res_cen = np.empty([len(vote_res_index), 11], dtype=object)
    res_index = 0

    with open('failure cases{}.csv'.format(filenum), 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['query_patch_name', 'q_center_row', 'q_center_col',
                             'target_patch_name', 'left', 'right', 'upper', 'lower', 'success_t_name', 'area'])

        for q_index in range(len(vote_res_index)):
            q_name = qname_csv[q_index + 1, 0]  # corresponding img name
            q_cen_row = int(q_center_row_csv[q_index + 1, 0])  # corresponding img position
            q_cen_col = int(q_center_col_csv[q_index + 1, 0])
            q_angle = int(angle_csv[q_index + 1, 0])
            vote_index = vote_res_index[q_index]
            if vote_index is None:
                res[res_index, :] = np.array([q_index, q_name, q_cen_row, q_cen_col, None, None, None, None, None, None, 0])
                # res_cen[res_index, :] = np.array([q_index, q_name, q_cen_row, q_cen_col, None, None, None, None, None, None, 0])

            else:
                for tile_idx in vote_index:  # loop each target tile index
                    t_name = tname_csv[tile_idx + 1, 0]  # target tile name
                    t_left = int(tleft_csv[tile_idx + 1, 0])
                    t_right = int(tright_csv[tile_idx + 1, 0])
                    t_upper = int(tupper_csv[tile_idx + 1, 0])
                    t_lower = int(tlower_csv[tile_idx + 1, 0])

                    success, area = is_overlap(tile_size, q_cen_row, q_cen_col, t_upper, t_lower, t_left, t_right, q_angle)

                    # success_cen = is_overlap_center(tile_size, q_cen_row, q_cen_col, t_upper, t_lower, t_left, t_right)

                    # if success != success_cen:  # retrieved images are not overlap but close
                    #     print(q_name, q_index, q_name, q_cen_row, q_cen_col, tile_idx, t_name, t_left, t_right, t_upper, t_lower, success)

                    res[res_index, :] = np.array([q_index, q_name, q_cen_row, q_cen_col, tile_idx, t_name, t_left, t_right, t_upper, t_lower, success])
                    # res_cen[res_index, :] = np.array([q_index, q_name, q_cen_row, q_cen_col, tile_idx, t_name, t_left, t_right, t_upper, t_lower, success_cen])

                    if success == 0:
                        filewriter.writerow([q_name, q_cen_row, q_cen_col, t_name, t_left, t_right, t_upper, t_lower, '_', area])
                    if success == 1:
                        filewriter.writerow([q_name, q_cen_row, q_cen_col, '_', t_left, t_right, t_upper, t_lower, t_name, area])

                if len(vote_index) < args.topk:
                    res[res_index:res_index + args.topk - len(vote_index), :] = np.tile([0], [args.topk - len(vote_index), 10])
                    # res_cen[res_index:res_index + args.topk - len(vote_index), :] = np.tile([0], [args.topk - len(vote_index), 10])

                    res_index = res_index + args.topk - len(vote_index)
            res_index = res_index + 1

    succ, fail = 0, 0
    for i in range(0, filenum + 1, args.topk):
        if 1 in res[i:i + args.topk, 10].astype(int):
            succ = succ + 1
        else:
            fail = fail + 1
    accuracy = succ / (fail + succ)
    print(args.query_data)
    print("============num of success:{}; num of fail:{}; acc:{}".format(succ, fail, accuracy))

    # succ, fail = 0, 0
    # for i in range(0, filenum + 1, args.topk):
    #     if 1 in res_cen[i:i + args.topk, 10].astype(int):
    #         succ = succ + 1
    #     else:
    #         fail = fail + 1
    #
    # accuracy = succ / (fail + succ)
    # print("============is overlap center num of success:{}; num of fail:{}; acc:{}".format(succ, fail, accuracy))
    return res, accuracy


def overlap_table(args, query_loader, target_loader, vote_res_index):
    query_n = len(query_loader)
    target_n = len(target_loader.dataset)

    # load metadata to list
    query_info = os.path.join(args.query_data, 'query_info.csv')
    target_info = os.path.join(args.target_data, 'target_info.csv')
    qname_csv, q_center_row_csv, q_center_col_csv, len_qinfo, angle_csv = load_info_center(query_info)
    tname_csv, tleft_csv, tright_csv, tupper_csv, tlower_csv, len_tinfo = load_info(target_info)

    overlap_dict = {'10': {'success': 0, 'fail': 0}, '20': {'success': 0, 'fail': 0}, '30': {'success': 0, 'fail': 0},
                    '40': {'success': 0, 'fail': 0}, '50': {'success': 0, 'fail': 0}, '60': {'success': 0, 'fail': 0},
                    '70': {'success': 0, 'fail': 0}, '80': {'success': 0, 'fail': 0}, '90': {'success': 0, 'fail': 0},
                    '100': {'success': 0, 'fail': 0}}

    with open('overlap_table.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['query_patch_name', 'q_center_row', 'q_center_col', 'angle', 'target_num',
                             'target0_index', 'target0_name', 'target0_area',
                             'target1_index', 'target1_name', 'target1_area',
                             'target2_index', 'target2_name', 'target2_area',
                             'target3_index', 'target3_name', 'target3_area',
                             'target4_index', 'target4_name', 'target4_area'])

        # read metadata to variables
        for q_index in range(query_n):
            print(q_index)
            q_name = qname_csv[q_index + 1, 0]  # corresponding img name
            q_cen_row = int(q_center_row_csv[q_index + 1, 0])  # corresponding img position
            q_cen_col = int(q_center_col_csv[q_index + 1, 0])
            q_angle = int(angle_csv[q_index + 1, 0])

            # vote_index = vote_res_index[q_index]
            temp = [q_name, q_cen_row, q_cen_col, q_angle, 0]
            target_num = 0
            for t_index in range(target_n):
                t_name = tname_csv[t_index + 1, 0]  # target tile name
                t_left = int(tleft_csv[t_index + 1, 0])
                t_right = int(tright_csv[t_index + 1, 0])
                t_upper = int(tupper_csv[t_index + 1, 0])
                t_lower = int(tlower_csv[t_index + 1, 0])

                success, area = is_overlap(TILE_SIZE, q_cen_row, q_cen_col, t_upper, t_lower, t_left, t_right, q_angle)

                if success == 1:  # this is the target image
                    temp.append(t_index)
                    temp.append(t_name)
                    temp.append(area)
                    target_num += 1
                #     lap_perc = round(area / TILE_SIZE ** 2) * 100
                #
                #     if t_index in vote_index:
                #         overlap_dict[str(lap_perc)]['success'] += 1
                #     else:
                #         overlap_dict[str(lap_perc)]['fail'] += 1
            temp[4] = target_num
            filewriter.writerow(temp)
    print(overlap_dict)


import csv
import argparse
import geolocate_utils
import glob
import os
from torchvision import transforms, datasets


def main():
    parser = argparse.ArgumentParser(description='perform rotation alignment and geo-localization')
    parser.add_argument('--query_data', help='path to query dataset')
    parser.add_argument('--target_data', help='path to target dataset')

    parser.add_argument('--input_size', default=224,
                        help='input size during geo-localization, pairs input:370, pairs created in Dataset, one input:224')

    # load dataset params
    args = parser.parse_args()
    args.query_data = '/hdd/wacv_2020/dataset/la_dataset/2012/tiles/370x370/random_query_800_2nd/tiles/rotrandom'
    args.target_data = '/hdd/wacv_2020/dataset/la_dataset/2014/tiles/256x256/target_set'

    transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_dataset = datasets.ImageFolder(root=args.target_data, transform=transform)
    query_dataset = datasets.ImageFolder(root=args.query_data, transform=transform)

    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=24, shuffle=False,
                                                num_workers=1, pin_memory=True)
    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=1, shuffle=False,
                                               num_workers=1, pin_memory=True)
    overlap_table(args, query_loader, target_loader, 'a')


# if __name__ == '__main__':
#     main()







