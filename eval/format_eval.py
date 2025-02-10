import os
from pathlib import Path
import json

def read_txt(path):
    with open(path, "r") as fin:
        data = fin.readline().strip()
    return data


def load_data(args, anno_path, split=None):
    '''
    anno data example:
    {"annotations":
        [
            {
                "image_id": "xHr8X2Wpmno.mp4"
                ...
            },
            ...
        ]
    }
    '''
    file_path = os.path.join(anno_path, f'{split}.caption_coco_format.json')
    with open(file_path, 'r') as f:
        data = json.load(f)["annotations"]

    if args.debug:
        data = data[:10]
    return data


def merge_seg_caps(results):
    """merge mulple generated captions from a same video into paragraph."""
    merge_results = {}
    for jterm in results:
        vname = jterm["vname"]
        cap = jterm["generated_cap"]
        postfix = vname.split(".mp4")[-1]
        start_time, end_time = float(postfix.split("_")[-2]), float(postfix.split("_")[-1])
        vid = vname.split(".mp4")[0] + ".mp4"
        if vid not in merge_results:
            merge_results[vid] = []
        merge_results[vid].append({"timestamp": [start_time, end_time], "caption": cap})
    return merge_results


def save_result(args, output_dir, results, split_name='test', format=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_name = f'{args.dataset}_{split_name}_clipF{args.infer_clip_frames}_result.json'
    if args.timestamp:
        if args.timestamp_file != '':
            file_name = f'{args.dataset}_{split_name}_clipF{args.infer_clip_frames}_result_with_pred_timestamp.json'
        else:
            file_name = f'{args.dataset}_{split_name}_clipF{args.infer_clip_frames}_result_with_gt_timestamp.json'
    if args.debug:
        file_name = 'debug_' + file_name
    if format:
        file_name = 'fmt_' + file_name
    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(results, f)
    return


def get_timestamp_from_file(timestamp_file):
    timestamp = {}
    with open(timestamp_file, 'r') as f:
        data = json.load(f)
        for vid, vlist in data.items():
            timestamp[vid] = []
            for vterm in vlist:
                timestamp[vid].append(vterm["timestamp"])
    return timestamp



