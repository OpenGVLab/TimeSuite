from configs.instruction_data import *

# ========================= data ==========================

train_corpus = "TimePro_Normal"

train_file = "${available_corpus[${train_corpus}]}"  
test_file = dict()
test_types = []
num_workers = 6

stop_key = None

# ========================= input ==========================
num_frames_test = 16
num_frames = 192
clip_frames = 8
good_init = True

batch_size = 2
max_txt_l = 1536

save_iter=1000

pre_text = False

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
)

# ========================= model ==========================
model = dict(
    model_cls="VideoChat2_it4_mistral_LinearP",
    vit_blip_model_path="/path_to_the_timesuite_root_folder/download/parameters/umt_l16_qformer.pth",
    mistral_model_path="/path_to_the_timesuite_root_folder/download/parameters/Mistral-7B-Instruct-v0.2",
    videochat2_model_path="/path_to_the_timesuite_root_folder/download/parameters/videochat2_mistral_7b_stage3.pth",
    freeze_vit=True,
    freeze_qformer=False,
    max_txt_len="${max_txt_l}", # use large max_txt_len on stage3
    clip_frames="${clip_frames}",
    num_frames="${num_frames}",
    token_merge_len = 4,
    # vit
    low_resource=False,
    vision_encoder=dict(
        name="vit_l14",
        img_size=224, 
        patch_size=16, 
        d_model=1024,
        encoder_embed_dim=1024, 
        encoder_depth=24,
        encoder_num_heads=16, 
        drop_path_rate=0., 
        num_frames="${clip_frames}",
        tubelet_size=1,
        use_checkpoint=True,
        checkpoint_num=18,
        pretrained="",
        return_index=-2,
        vit_add_ln=True,
        ckpt_num_frame=4,
    ),
    # qformer
    num_query_token=32,
    qformer_hidden_dropout_prob=0.1,
    qformer_attention_probs_dropout_prob=0.1,
    qformer_drop_path_rate=0.2,
    extra_num_query_token=64,
    qformer_text_input=True,
    # prompt
    system="",
    start_token="<Video>",
    end_token="</Video>",
    add_second_msg=True,
    img_start_token="<Image>", 
    img_end_token="</Image>",
    random_shuffle=True, 
    return_question_instruction=False,
    use_flash_attention=True,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    # debug=True,
)

optimizer = dict(
    opt="adamW",
    lr=2e-5,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.25, warmup_epochs=0.6)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

fp16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="likunchang",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="videogpt",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "it_mistral"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 10
seed = 42

save_latest = False
auto_resume = True
pretrained_path = ""  # path to pretrained model weights, for resume only?

deepspeed = dict(
    enable=False,
    stage=1,
)