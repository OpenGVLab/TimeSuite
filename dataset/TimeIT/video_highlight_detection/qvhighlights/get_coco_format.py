import json
import argparse
import os
from copy import deepcopy
import pdb
import numpy as np
import random 
from pathlib import Path
from collections import Counter

# read json files
def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
        annos = datas["annotations"]
    return annos


def read_jsonl(path):
    anno = []
    with open(path, "r") as fin:
        datas = fin.readlines()
        for data in datas:
            anno.append(json.loads(data.strip()))
    return anno



def write_json(data, path):
    with open(path, "w") as fout:
        json.dump(data, fout)
    return


def read_txt(path):
    data = []
    with open(path, "r") as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            # e.g. AO8RW 0.0 6.9##a person is putting a book on a shelf.
            line = line.strip("\n")
            cap = line.split("##")[-1]
            if len(cap) < 2:
                continue
            terms = line.split("##")[0].split(" ")
            vid = terms[0] + ".mp4"
            start_time = float(terms[1])
            end_time = float(terms[2])
            data.append({"image_id": vid, "caption": cap, "timestamp": [start_time, end_time], "id": i})
    return data


def filter_sent(sent):
    sent = sent.strip(" ")
    if len(sent) < 2:
        return False
    sent = sent.replace("#", "")
    return sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='qvhighlights') # anet
    parser.add_argument('--anno_path', default='annotations_raw/')
    parser.add_argument('--video_path', default='videos/') # ActivityNet_asr_denseCap/anet_6fps_224
    parser.add_argument('--outpath', default='./')
    args = parser.parse_args()
    '''output data example:
    {
        "annotations": [ 
        {   
            "image_id": "3MSZA.mp4", 
            "caption": "person turn a light on.",
            "timestamp": [24.3, 30.4],
        }],
    }
    '''
    miss_videos = []
    num_clips = []
    for split in ["train", "val"]: # "val", "test"
        if args.dataset == "charades":
            filename = f"charades_sta_{split}.txt"
            annos = read_txt(os.path.join(args.anno_path, filename))
            data = {}
            data["annotations"] = annos
        elif args.dataset == "qvhighlights":
            filename = f"highlight_{split}_release.jsonl"
            annos = read_jsonl(os.path.join(args.anno_path, filename))
            new_data = []
            for jterm in annos:
                new_term = {}
                new_term["image_id"] = "v_" + jterm["vid"] + ".mp4"
                # check the existance of the video
                if not os.path.exists(os.path.join(args.video_path, split, new_term["image_id"])):
                    miss_videos.append(new_term["image_id"])
                    continue
                new_term["id"] = jterm["qid"]
                new_term["caption"] = jterm["query"]
                new_term["timestamp"] = jterm["relevant_windows"]
                new_term["duration"] = jterm["duration"]
                new_term["relevant_clip_ids"] = jterm["relevant_clip_ids"]
                new_term["saliency_scores"] = jterm["saliency_scores"]
                new_data.append(new_term)
                num_clips.append(int(jterm["duration"]/2))
            data = {}
            data["annotations"] = new_data
        else:
            print("Do not support this dataset!")
            exit(0)
            
        print(f"==> {args.dataset} dataset  \t# examples num: {len(new_data)} \t# miss videos num: {len(miss_videos)}\t# raw data num: {len(annos)}")
        out_name = "{}.caption_coco_format.json".format(split)
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        write_json(data, os.path.join(args.outpath, out_name))
        
        if len(num_clips) >= 1:
            count = Counter(num_clips)
            # sort count dict with the clip num
            print(count)
            print(max(list(count.keys())))
            
