import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

anno_root_it = "/path_to_the_timesuite_root_folder/download/datasets/TimePro"


# ============== pretraining datasets=================
available_corpus = dict(
    
    caption_youcook2=[
        f"{anno_root_it}/caption_youcook2.json", 
        "pnorm2:s3://youcook2/split_videos",
        "video"
    ],
    conversation_videochat1=[
        f"{anno_root_it}/conversation_videochat1.json", 
        "pnorm2:s3://webvid10m",
        "video"
    ],
    conversation_videochat2=[
        f"{anno_root_it}/conversation_videochat2.json", 
        "pnorm:s3://videointernsegvideos",
        "video"
    ],
    conversation_videochatgpt=[
        f"{anno_root_it}/conversation_videochatgpt.json", 
        "pnorm2:s3://anet/ANet_320p_fps30",
        "video"
    ],
    reasoning_star=[
        f"{anno_root_it}/reasoning_star.json", 
        "pnorm2:s3://star/Charades_v1_480",
        "video"
    ],
    vqa_ego_qa=[
        f"{anno_root_it}/vqa_ego_qa.json", 
        "pnorm2:s3://egoqa/split_videos",
        "video"
    ],




    # TimeIT
    timeit_ANet=[
        f"{anno_root_it}/timeit_ANet.json", 
        "pnorm2:s3://anet",
        "video"
    ],
    
    timeit_COIN=[
        f"{anno_root_it}/timeit_COIN.json", 
        "pnorm:s3://COIN_320p",
        "video"
    ],

    timeit_DiDeMo=[
        f"{anno_root_it}/timeit_DiDeMo.json", 
        "sssd:s3://yjsBucket",
        "video"
    ],
    
    timeit_HiREST=[
        f"{anno_root_it}/timeit_HiREST.json", 
        "pnorm2zxy:s3://hirest",
        "video"
    ],
    
    
    timeit_QuerYD=[
        f"{anno_root_it}/timeit_QuerYD.json", 
        "pnorm2zxy:s3://queryd",
        "video"
    ],
    
    timeit_TVSum=[
        f"{anno_root_it}/timeit_TVSum.json", 
        "pnorm2zxy:s3://tvsum",
        "video"
    ],
    
    timeit_ViTT=[
        f"{anno_root_it}/timeit_ViTT.json", 
        "sssd:s3://ViTT",
        "video"
    ],
    
    timeit_yttemporal180m=[
        f"{anno_root_it}/timeit_yttemporal180m.json", 
        "pnorm:s3://YT-Temporal-180M",
        "video"
    ],
    
    grounding_ANetRTL=[    
        f"{anno_root_it}/grounding_ANetRTL.json", 
        "pnorm2:s3://anet/ANet_320p_fps30/train",
        "video"
    ],
    
    grounding_IntrenvidVTime_100K=[
        f"{anno_root_it}/grounding_IntrenvidVTime_100K.json", 
        "pnorm:s3://youtubeBucket/videos/",
        "video"
    ],
    grounding_ANetHL2=[
        f"{anno_root_it}/grounding_ANetHL2.json", 
        "pnorm2:s3://anet/ANet_320p_fps30/train",
        "video"
    ],

    grounding_CosmoCap_93K=[
        f"{anno_root_it}/grounding_CosmoCap_93K.json", 
        "pvideo:s3://howto100m/",
        "video"
    ],
    vqa_moviechat = [
        f'{anno_root_it}/vqa_moviechat.json',
        'pnorm2:s3://MovieChat/real_video/',
        'video'
    ],
    caption_moviechat = [
        f'{anno_root_it}/caption_moviechat.json',
        'pnorm2:s3://MovieChat/real_video/',
        'video'
    ],
    
    
    FT_Charades=[
        f"{anno_root_it}/FT_Charades.json", 
        "s3://zengxiangyu/Charades/",
        "video"
    ],
    
    FT_QVHighlights=[
        f"{anno_root_it}/FT_QVHighlights.json", 
        "s3://QVHighlight/videos/",
        "video"
    ],
    
)


available_corpus["TimePro_Normal"] = [    #final dataset
    #TiIT
    available_corpus["timeit_ANet"],        
    available_corpus["timeit_COIN"],        
    available_corpus["timeit_DiDeMo"],      
    available_corpus["timeit_HiREST"],      
    available_corpus["timeit_QuerYD"],      
    available_corpus["timeit_TVSum"],       
    available_corpus["timeit_ViTT"],        
    available_corpus["timeit_yttemporal180m"],     
    #Conv
    available_corpus["conversation_videochatgpt"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochat1"],
    #DvcVqa
    available_corpus["caption_youcook2"],
    available_corpus["vqa_ego_qa"],
    #Gro
    available_corpus["grounding_ANetRTL"],
    available_corpus["grounding_IntrenvidVTime_100K"],
    available_corpus["grounding_ANetHL2"],
    available_corpus["grounding_CosmoCap_93K"],
    available_corpus["vqa_moviechat"],
    available_corpus["caption_moviechat"],
    available_corpus["reasoning_star"],
]



available_corpus["FT_Temporal_Grounding_Both"] = [
    available_corpus["FT_Charades"],    
    available_corpus["FT_QVHighlights"],
    available_corpus["grounding_ANetHL2"],
    available_corpus["caption_youcook2"],
]