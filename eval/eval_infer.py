# In[1]:
import re
import os
import sys
import math
# 获取当前.py文件的绝对路径
file_path = os.path.abspath(__file__)
# 获取.py文件所在的上级的上级目录
grandparent_dir = os.path.dirname(os.path.dirname(file_path))
# 将上级的上级目录添加到sys.path中
sys.path.append(grandparent_dir)
# 打印sys.path以确认路径已被添加
print(sys.path)

import copy
import torch
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList
from math import ceil
from PIL import Image
import numpy as np
import decord

decord.bridge.set_bridge('torch')
from torchvision.transforms.functional import InterpolationMode
import json
import time
import datetime
from tqdm import tqdm
import random

random.seed(1234)
from eval.format_dvc import format_dvc
from eval.format_tvg import format_tvg
from eval.format_vhd import format_vhd
from eval.format_eval import *


from utils.config import Config
from utils.easydict import EasyDict
from transformers import StoppingCriteria, StoppingCriteriaList
from decord import VideoReader, cpu
import torchvision.transforms as T
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from peft import get_peft_model, LoraConfig, TaskType
from io import BytesIO
from models import *

try:
    from petrel_client.client import Client
    has_client = True
    print("Client on!")
except:
    has_client = False
    print("Client off!")

if has_client:
    client = Client('~/petreloss.conf')
else:
    client = None


# In[2]:


def get_args():
    parser = argparse.ArgumentParser()
    #与测试任务无关
    parser.add_argument('--model_type', default="Please input model type!")
    parser.add_argument('--model_dir', default="Please input model dir!")
    parser.add_argument('--model_pth', default="Please input model pth!")
    parser.add_argument('--output_dir', default="Please input model output dir!")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--infer_clip_frames', type=int, default=8)
    
    #与测试任务有关
    parser.add_argument('--task', required=True, default='Please input task!')  # dvc for dense video captioning; tvg for temporal video grounding; vhd for video highlight detection
    parser.add_argument('--split', required=True, default='Please input split!')
    parser.add_argument('--dataset', required=True, default='Please input dataset!')
    parser.add_argument('--prompt_file', required=True, default='Please input prompt dir!')
    parser.add_argument('--anno_path', required=True, type=str, default='Please input anno dir!')
    parser.add_argument('--video_path', required=True, type=str, default='Please input video dir!')
    parser.add_argument('--post_processing_vhd', action='store_true', help='post_processing_vhd')
    
    
    #NOTE：以下参数是原版带的，懒得删干净了，可以不管
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--timestamp', action='store_true', help='input the gt/predicted timestamps to the model')
    parser.add_argument('--timestamp_file', type=str, default='', help='the predcited timestamps file')
    parser.add_argument('--debug', action='store_true', help='the debug mode will only use 10 data samples')
    parser.add_argument('--sample_num', type=int, default=-1, help='fast inference by sampling N instances to evaluate')
    
    args = parser.parse_args()    
    return args


args = get_args()
args_list_str = '\n' + '\n'.join([f'{k:<25}: {v}' for k, v in vars(args).items()])
print(args_list_str)



# load data
video_path = args.video_path
anno_path = args.anno_path
print("anno_path: ", anno_path)
anno_data = load_data(args, anno_path, split=args.split)
if args.timestamp_file != '':
    pred_timestamps = get_timestamp_from_file(args.timestamp_file)
vids = []
vnames = []
captions = []
qids = []

if args.sample_num > 0:
    # sample part data to evaluate
    anno_data = random.sample(anno_data, args.sample_num)
    
for jterm in anno_data:
    vname = jterm["image_id"]
    vid_path = os.path.join(video_path, vname)
    if args.timestamp:
        duration = int(jterm["duration"])
        if args.timestamp_file == '':  # input the gt timestamps
            timestamp = jterm["segments"]
        else:  # input the pred timestamps
            timestamp = pred_timestamps[vname]
        for (start_time, end_time) in timestamp:
            # process anno timestamp error
            if start_time >= end_time or end_time > duration or start_time >= duration:
                continue
            vids.append(vid_path)
            vnames.append(vname + "_" + str(start_time) + "_" + str(end_time))
            # image_emb, _ = model.encode_img(video)
            # img_lists.append([image_emb])
    else:
        vids.append(vid_path)
        vnames.append(vname)
        captions.append(jterm["caption"])
        qids.append(jterm["id"])

prompt = read_txt(args.prompt_file)
eval_start_time = time.time()
print('Dataset Initialization Finished')



# In[5]:

# config_file = "configs/config_mistral.json"
config_file = args.model_dir+"/config.json"

cfg = Config.from_file(config_file)
cfg.model.use_lora = False


print("vision_encoder.num_frames:", cfg.model.vision_encoder.num_frames)
# cfg.model.vision_encoder.num_frames = 4

model_cls = eval(args.model_type)
# model = VideoChat2_it_mistral(config=cfg.model)
model = model_cls(config=cfg.model)


# add lora to run stage3 model
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
    r=16, lora_alpha=32, lora_dropout=0.,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj", "lm_head"
    ]
)
model.mistral_model = get_peft_model(model.mistral_model, peft_config)


# state_dict = torch.load("/path_to_the_timesuite_root_folder/download/parameters/videochat2_mistral_7b_stage3.pth", "cpu")
state_dict = torch.load(args.model_dir+"/"+args.model_pth+".pth", "cpu")


if 'model' in state_dict.keys():
    msg = model.load_state_dict(state_dict['model'], strict=False)
else:
    msg = model.load_state_dict(state_dict, strict=False)
print(msg)

model = model.to(torch.device(cfg.device))
model = model.eval()

print('Model Initialization Finished')



def post_processing_vhd(text):    
    # 提取时间段和显著性分数
    match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*).*?(\d+\.?\d*)', text)
    if not match:
        return "Invalid post processing format! " + text
    
    start, end, score = float(match.group(1)), float(match.group(2)), float(match.group(3))
    start = math.floor(start)
    end = math.ceil(end)
    # 生成离散时间戳和显著性分数
    timestamps = [round(start + i * 2) for i in range(int((end - start) / 2) + 1)]
    scores = [score] * len(timestamps)
    
    # 构建转换后的文本
    new_text = f"The highlight timestamps are in the {', '.join(map(str, timestamps))} seconds. Their saliency scores are {', '.join(map(str, scores))}."
    
    return new_text



def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + " " + message + " " + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + " " + message
        else:
            if message:
                ret += role + " " + message + " " + conv.sep
            else:
                ret += role
    return ret


def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print("prompt:",prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.mistral_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
#         seg_embs = [model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text])
        

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([2]).to("cuda:0"),
        torch.tensor([29871, 2]).to("cuda:0")]  # '</s>' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.mistral_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
#     output_text = output_text.split('[/INST]')[-1].strip()
    conv.messages[-1][1] = output_text + '</s>'
    return output_text, output_token.cpu().numpy()



def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224):
    
    if client is not None and "s3" in video_path:
        video_bytes = client.get(video_path)
        assert(video_bytes is not None)
        vr = VideoReader(BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
    else:
        vr = VideoReader(uri=video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds. "
        return torch_imgs, msg
    else:
        return torch_imgs
    

# In[8]:


def generate_videochat(vid_path, user_messages):
    
    num_frame = model.clip_frames
    tot_frames = model.total_frames
    resolution = cfg.model.vision_encoder.img_size
    
    vid, msg = load_video(vid_path, num_segments=tot_frames, return_msg=True, resolution=resolution)

    # The model expects inputs of shape: T x C x H x W
    TC, H, W = vid.shape
    video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")

    img_list = []
    with torch.no_grad():
        image_emb = model.encode_long_video(video,[msg,],"")
        print("Shape of long video embeds: ", image_emb.shape)
#         image_emb, _ = model.encode_img(video, "")
    img_list.append(image_emb)
    
    chat = EasyDict({
    "system": "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail. ",
    "roles": ("[INST]", "[/INST]"),
    "messages": [],
    "sep": ""
    })
    
    chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
    ask(msg+user_messages, chat)

    llm_answer = answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=512, print_res=True)[0]
    
    if args.post_processing_vhd:
        llm_answer = post_processing_vhd(llm_answer)
    
    print("LLM answer:", llm_answer,"\n\n\n")
    
    return llm_answer, chat, img_list
    
    


# In[10]:


results = []
bz = args.batch_size
# evaluate using batch
epoch = ceil(len(vnames) / bz)
for i in tqdm(range(epoch)):
    sid = i * bz
    eid = min((i + 1) * bz, len(vnames))
    prompts = []
    # load video
    paths = vids[sid:eid]
    image_ids = qids[sid:eid]
    for pi in range(len(paths)):
        final_prompt = copy.deepcopy(prompt)
        #NOTE:delete asr processing from here
        if args.task in ["tvg", "vhd"]:
            idx = sid + pi
            prompts.append(final_prompt.format(args.dataset, captions[idx].strip('.')))
        else:
            prompts.append(final_prompt)
    
    
    outputs, chat_states, img_lists = generate_videochat(paths[0], prompts[0])
    

    for j, (output, chat_state) in enumerate(zip([outputs], [chat_states])):
        if args.task in ["tvg", "vhd"]:
            results.append({
                "vname": vnames[sid + j],
                "generated_cap": output,
                "query": captions[sid + j],
                "id": qids[sid + j],
                "prompt": chat_state
            })
        else:
            results.append({
                "vname": vnames[sid + j],
                "generated_cap": output,
                "prompt": chat_state
            })
        # with open(output_file, 'a') as f:
        #     print(json.dumps(results[-1]), file=f, flush=True)

if args.timestamp:
    results = merge_seg_caps(results)
    
# save results
save_result(args, args.output_dir, results, args.split)

# format results to calculate metrics
if args.task == "dvc":
    fmt_results = format_dvc(results)
elif args.task == "tvg":
    fmt_results = format_tvg(results)
elif args.task == "vhd":
    fmt_results = format_vhd(results, anno_data)
else:
    print(f"Not support formatting samples for task {args.task}")
    
# save format results
save_result(args, args.output_dir, fmt_results, args.split, format=True)


total_time = time.time() - eval_start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Evaluate time {}'.format(total_time_str))


with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
    f.write(args_list_str + "\n")
    f.write(json.dumps(cfg, indent=4) + "\n")
print("Done!")