import copy
import os
from argparse import ArgumentParser
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def add_sub(df_q, fig, index=1, all=4, col_name='dif_x1'):
    # 1 of 4
    ax1 = fig.add_subplot((all+3)//4, 4, index) # 1 row x 4 col, set 1
    ax1.set_ylabel('frequency')
    ax1.set_title(col_name)
    ax1.grid(axis='y', color='gray', lw=0.5)

    n, bins, _ = plt.hist(df_q[col_name], bins=20)
    xs = (bins[:-1] + bins[1:])/2 # 各柱の端が返るのでずらす
    ys = n
    for x, y in zip(xs, ys):
        if y > 0:
            plt.text(x, y, str(int(y)), horizontalalignment="center")
    return

def add_sub_hv_stack(df_q, fig, index=1, all=4, col_name='dif_x1'):
    # 1 of 4
    ax1 = fig.add_subplot((all+3)//4, min(all, 4), index) # 1 row x 4 col, set 1
    ax1.set_ylabel('frequency')
    ax1.set_title(col_name)
    ax1.grid(axis='y', color='gray', lw=0.5)

    n, bins, _ = plt.hist(df_q[col_name], bins=20, color='C0', label='all')
    n_h, _, _  = plt.hist(df_q.query('gt_w > gt_h')[col_name],
                            histtype='stepfilled', color='C1', bins=bins, label='hori')
    ax1.legend()
    xs = (bins[:-1] + bins[1:])/2 # 各柱の端が返るのでずらす
    ys = n
    for x, y in zip(xs, ys):
        if y > 0:
            plt.text(x, y, str(int(y)), horizontalalignment="center")
    ys = n_h
    for x, y in zip(xs, ys):
        if y > 0:
            plt.text(x, y, str(int(y)), horizontalalignment="center", color='C1')
    return

def save_hist(df, query, col_names, save_dir='hist_png', fname_head='', hv_stack=False):
    df_q = df.query(query)

    fig = plt.figure(figsize=(8.0*min(len(col_names),4), 6.0*((len(col_names)+3)//4)), facecolor="azure", edgecolor="coral")
    fig.suptitle(query)

    if hv_stack:
        for i in range(len(col_names)):
            add_sub_hv_stack(df_q, fig, i+1, len(col_names), col_names[i])
    else:
        for i in range(len(col_names)):
            add_sub(df_q, fig, i+1, len(col_names), col_names[i])

    savepng_name = '{}.png'.format(fname_head)
    print('Save: {}'.format(savepng_name))
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, savepng_name), bbox_inches='tight')
    plt.clf()
    plt.close()
    return

def analyze_individual_category(k,
                                cocoDt,
                                cocoGt, 
                                catId,
                                iou_type,
                                areas=None):
    nm = cocoGt.loadCats(catId)[0]
    print(f'--------------analyzing {k + 1}-{nm["name"]}---------------')
    ps_ = {}
    dt = copy.deepcopy(cocoDt)
    nm = cocoGt.loadCats(catId)[0]
    imgIds = cocoGt.getImgIds()
    dt_anns = dt.dataset['annotations']
    select_dt_anns = []
    for ann in dt_anns:
        if ann['category_id'] == catId:
            select_dt_anns.append(ann)
    dt.dataset['annotations'] = select_dt_anns
    dt.createIndex()
    # compute precision but ignore superclass confusion
    gt = copy.deepcopy(cocoGt)
    if nm.get('supercategory'):
        child_catIds = gt.getCatIds(supNms=[nm['supercategory']])
        for idx, ann in enumerate(gt.dataset['annotations']):
            if ann['category_id'] in child_catIds and ann['category_id'] != catId:
                gt.dataset['annotations'][idx]['ignore'] = 1
                gt.dataset['annotations'][idx]['iscrowd'] = 1
                gt.dataset['annotations'][idx]['category_id'] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [100]
    cocoEval.params.iouThrs = [0.1]
    cocoEval.params.useCats = 1
    if areas:
        cocoEval.params.areaRng = [[0**2, areas[2]], [0**2, areas[0]],
                                   [areas[0], areas[1]], [areas[1], areas[2]]]
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_supercategory = cocoEval.eval['precision'][0, :, k, :, :]
    ps_['ps_supercategory'] = ps_supercategory
    # compute precision but ignore any class confusion
    gt = copy.deepcopy(cocoGt)
    for idx, ann in enumerate(gt.dataset['annotations']):
        if ann['category_id'] != catId:
            gt.dataset['annotations'][idx]['ignore'] = 1
            gt.dataset['annotations'][idx]['iscrowd'] = 1
            gt.dataset['annotations'][idx]['category_id'] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [100]
    cocoEval.params.iouThrs = [0.1]
    cocoEval.params.useCats = 1
    if areas:
        cocoEval.params.areaRng = [[0**2, areas[2]], [0**2, areas[0]],
                                   [areas[0], areas[1]], [areas[1], areas[2]]]
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_allcategory = cocoEval.eval['precision'][0, :, k, :, :]
    ps_['ps_allcategory'] = ps_allcategory
    return k, ps_

def analyze_results(res_file,
                    ann_file,
                    res_types,
                    out_dir,
                    out_csv,
                    hv = 'SUM',
                    histplots=None,
                    areas=None):
    for res_type in res_types:
        assert res_type in ['bbox', 'segm']
    if areas:
        assert len(areas) == 3, '3 integers should be specified as areas, \
            representing 3 area regions'

    directory = os.path.dirname(out_dir + '/')
    if not os.path.exists(directory):
        print(f'-------------create {out_dir}-----------------')
        os.makedirs(directory)

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)
    imgIds = cocoGt.getImgIds()
    for res_type in res_types:
        iou_type = res_type
        cocoEval = COCOeval(
            copy.deepcopy(cocoGt), copy.deepcopy(cocoDt), iou_type)
        cocoEval.params.imgIds = imgIds
        cocoEval.params.iouThrs = [0.75, 0.5, 0.1]
        cocoEval.params.maxDets = [100]
        if areas:
            cocoEval.params.areaRng = [[0**2, areas[2]], [0**2, areas[0]],
                                       [areas[0], areas[1]],
                                       [areas[1], areas[2]]]
        cocoEval.evaluate()
        cocoEval.accumulate() # ここまでで解析実行完了

        print("=========================")
        cols = ['file_name', 'image_id', 'gt_id', 'dt_id', 'category_id', 'iou', 'score',
                'gt_x', 'gt_y', 'gt_w', 'gt_h', # GT
                'dt_x', 'dt_y', 'dt_w', 'dt_h', # Detected bbox of max iou
                'dif_x1', 'dif_y1', 'dif_x2', 'dif_y2', 'dif_w', 'dif_h', # dt - gt
                'rat_x1', 'rat_y1', 'rat_x2', 'rat_y2', 'rat_w', 'rat_h' # (dt - gt) / gt_w(or gt_h)
                ]
        df_pos = pd.DataFrame(index=[], columns=cols)
        for imgId in cocoEval.params.imgIds:
            img_file_name = cocoGt.imgs[imgId]['file_name']
            print("processing {}: {}/{}".format(img_file_name, imgId+1, len(cocoEval.params.imgIds)))
            # img {'file_name': '1883229_R0000227_contents_L.jpg', 'width': 2602, 'height': 3329, 'id': 0}
            for catId in cocoEval.params.catIds:
                for dt_idx, iou_arr in enumerate(cocoEval.ious[(imgId, catId)]):
                    # print("idx: {}, iou_list:{}".format(dt_idx, iou_arr))
                    arg_max_idx =np.argmax(iou_arr)
                    gt_idx = cocoEval._gts[imgId, catId][arg_max_idx]['id']
                    # _gts[ImgId, CatId]で、ImgId中のCatIdのannがとれる
                    score = cocoEval._dts[imgId, catId][dt_idx]['score']
                    gt_bbox = cocoEval._gts[imgId, catId][arg_max_idx]['bbox'] # [x, y, w, h]
                    dt_bbox = cocoEval._dts[imgId, catId][dt_idx]['bbox']

                    dif_x1 = dt_bbox[0] - gt_bbox[0]
                    dif_y1 = dt_bbox[1] - gt_bbox[1]
                    dif_w  = dt_bbox[2] - gt_bbox[2]
                    dif_h  = dt_bbox[3] - gt_bbox[3]
                    dif_x2 = dif_x1 + dif_w      # right
                    dif_y2 = dif_y1 + dif_h      # bottom
                    rat_x1 = dif_x1 / gt_bbox[2] # dif_x / gt_w
                    rat_y1 = dif_y1 / gt_bbox[3] # dif_y / gt_h
                    rat_x2 = dif_x2 / gt_bbox[2] # dif_x / gt_w
                    rat_y2 = dif_y2 / gt_bbox[3] # dif_y / gt_h
                    rat_w  = dif_w  / gt_bbox[2] # dif_w / gt_w
                    rat_h  = dif_h  / gt_bbox[3] # idf_h / gt_h

                    record = pd.Series(
                        np.concatenate([
                            [img_file_name, imgId, gt_idx, dt_idx, catId, iou_arr[arg_max_idx], score],
                            gt_bbox,
                            dt_bbox,
                            [dif_x1, dif_y1, dif_x2, dif_y2, dif_w, dif_h,
                            rat_x1, rat_y1, rat_x2, rat_y2, rat_w, rat_h]],
                            axis=0) # concat
                        , index=df_pos.columns)
                    df_pos = df_pos.append(record, ignore_index=True)

        df_pos.to_csv(os.path.join(out_dir, out_csv))

        # create histograms
        df = pd.read_csv(os.path.join(out_dir, out_csv), index_col=0, header=0)
        if histplots:
            q_iou_list=['0.50<=iou',
                        '0.50<=iou<0.75',
                        '0.75<=iou<0.90',
                        '0.90<=iou']
            classes =  ['line_main' , 'line_inote', 'line_hnote', 'line_caption',
                        'block_fig', 'block_table', 'block_pillar', 'block_folio',
                        'block_rubi', 'block_chart', 'block_eqn', 'block_cfm',
                        'block_eng']

            for cid, cname in enumerate(classes):
                for q_iou in q_iou_list:
                    if hv=='SEPARATE':
                        # horizontally bbox
                        query = 'gt_w > gt_h & category_id=={} & {}'.format(cid, q_iou)
                        head  = '{}_{}_hori'.format(cname, q_iou)
                        col_names = ['dif_x1', 'dif_x2', 'dif_y1', 'dif_y2',
                                    'rat_x1', 'rat_x2', 'rat_y1', 'rat_y2',
                                    'dif_h', 'dif_w', 'rat_h', 'rat_w']
                        hist_save_dir = os.path.join(out_dir, 'hist_png')
                        save_hist(df, query, col_names, save_dir=hist_save_dir, fname_head=head)

                        # vertically bbox
                        query = 'gt_w < gt_h & category_id=={} & {}'.format(cid, q_iou)
                        head  = '{}_{}_vert'.format(cname, q_iou)
                        save_hist(df, query, col_names, save_dir=hist_save_dir, fname_head=head)
                    elif hv=='STACK':
                        query = 'category_id=={} & {}'.format(cid, q_iou)
                        head  = '{}_{}'.format(cname, q_iou)
                        col_names = ['dif_x1', 'dif_x2', 'dif_y1', 'dif_y2',
                                    'rat_x1', 'rat_x2', 'rat_y1', 'rat_y2',
                                    'dif_h', 'dif_w', 'rat_h', 'rat_w']
                        hist_save_dir = os.path.join(out_dir, 'hist_png')
                        save_hist(df, query, col_names, save_dir=hist_save_dir, fname_head=head, hv_stack=True)
                    else:
                        query = 'category_id=={} & {}'.format(cid, q_iou)
                        head  = '{}_{}'.format(cname, q_iou)
                        col_names = ['dif_x1', 'dif_x2', 'dif_y1', 'dif_y2',
                                    'rat_x1', 'rat_x2', 'rat_y1', 'rat_y2',
                                    'dif_h', 'dif_w', 'rat_h', 'rat_w']
                        hist_save_dir = os.path.join(out_dir, 'hist_png')
                        save_hist(df, query, col_names, save_dir=hist_save_dir, fname_head=head)
    return

def main():
    parser = ArgumentParser(description='COCO Error Analysis Tool')
    parser.add_argument('result', help='result file (json format) path')
    parser.add_argument('ann', help='annotation file (json format) path')
    parser.add_argument(
        '--out_dir',
        default='res_pos_analysis',
        help='output dir')
    parser.add_argument(
        '--out_csv',
        default='df_pos.csv',
        help='file to save analyze result csv')
    parser.add_argument(
        '--types', type=str, nargs='+', default=['bbox'], help='result types')
    parser.add_argument(
        '--hv',
        default='SUM',
        help='Create histograms with/without distinction between vertically and horizontally written documents.'
             'SUM(default): wihout distinction'
             'SEPARATE    : with distinction creating different bar charts'
             'STACK       : with distinction using one stacked bar charts'
    )
    parser.add_argument(
        '--histplots',
        action='store_false',
        help='export histogram plots (default is true')
    parser.add_argument(
        '--areas',
        type=int,
        nargs='+',
        default=[1024, 9216, 10000000000],
        help='area regions')
    args = parser.parse_args()
    analyze_results(
        args.result,
        args.ann,
        args.types,
        out_dir=args.out_dir,
        out_csv=args.out_csv,
        hv =args.hv,
        histplots=args.histplots,
        areas=args.areas)

if __name__ == '__main__':
    main()
