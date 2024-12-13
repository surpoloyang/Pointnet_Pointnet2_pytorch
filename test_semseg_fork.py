"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.PSDataLoader import PSDataset
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['sem', 'leave']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in testing [default: 128]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    # parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    # parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/PS_sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 2
    BATCH_SIZE = args.batch_size
    # NUM_POINT = args.num_point

    root = 'data/PepperSeedlings/test.txt'

    TEST_DATASET = PSDataset(data_root=root, transform=None)
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        # Load test data
        test_data_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        log_string('---- EVALUATION WHOLE SCENE----')

        # Initialize metrics
        total_correct = 0
        total_seen = 0
        iou_list = []
        total_seen_class = [0, 0]
        total_correct_class = [0, 0]
        total_iou_deno_class = [0, 0]
        if args.visual:
            # for i in range(len(test_data_loader)):
            fout = open(os.path.join(visual_dir, 'semseg_test_pred.txt'), 'w')
            fout_gt = open(os.path.join(visual_dir, 'semseg_test_gt_.txt' ), 'w')
        # Iterate over test data
        for points, target in tqdm(test_data_loader, total=len(test_data_loader)):
            points = points.float().cuda()
            target = target.long().cuda()
            points = points.transpose(2, 1)
            
            # Forward pass
            pred, _ = classifier(points)
            pred_choice = pred.data.max(2)[1]
            
            # Calculate metrics
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_seen += BATCH_SIZE
            
            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum(target.cpu().numpy() == l)
                total_correct_class[l] += np.sum((pred_choice.cpu().numpy() == l) & (target.cpu().numpy() == l))
                total_iou_deno_class[l] += np.sum((pred_choice.cpu().numpy() == l) | (target.cpu().numpy() == l))

            # Save visual results
            for i in range(points.size()[0]):
                color = g_label2color[pred_choice.reshape(-1)[i].item()]
                color_gt = g_label2color[target.reshape(-1)[i].item()]
                fout.write('%f %f %f %d %d %d\n' % (points[i, 0, 0], points[i, 1, 0], points[i, 2, 0], color[0], color[1], color[2]))
                fout_gt.write('%f %f %f %d %d %d\n' % (points[i, 0, 0], points[i, 1, 0], points[i, 2, 0], color_gt[0], color_gt[1], color_gt[2]))
        fout.close()
        fout_gt.close()
        # Calculate IoU
        iou_list = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6)
        mean_iou = np.mean(iou_list)

        # Log results
        log_string('eval point avg class IoU: %f' % mean_iou)
        log_string('eval point avg class acc: %f' % np.mean(np.array(total_correct_class) / np.array(total_seen_class)))
        log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))

        log_string('Done!')
        # print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
