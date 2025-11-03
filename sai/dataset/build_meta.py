""" Collect metadata for dataset, save as csv files
"""
import gc
import os
import os.path as osp
import json
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode


def read_captions(caption_file_path):
    caption_dict = {}
    #'{"fid": "23e9pbVQ_lo_1920x1080_full_video_001_A_00", "caption": The person in the video is sitting on a red sofa, holding a piece of paper in their right hand and placing their left hand on their knee. His right leg is bent while his left leg is straight. He appears to be reading the paper in his hand and occasionally nods his head, indicating agreement or understanding with the content. His head slightly turns to the left, looking at the paper, then returns to the front. Throughout the process, his body remains relatively stable, with only minor movements in his head and hands. The person says, <S> And Nicole says, OK, Ellen, you asked for something that would freak your freak, and this is what you've been waiting for. This is the As Seen on TV hat. Now you can watch videos on your iPod.<E> <AUDCAP>A female speaker with a medium pitch and a North American accent speaks at a fast, energetic rate. Her prosody is animated and expressive, characterized by clear stress on key words and a dynamic, engaging intonation that rises and falls. She projects enthusiasm and excitement through her vocal delivery, creating a lively and persuasive tone. The acoustic qualities suggest a younger adult speaker in a clear, broadcast-like recording.<ENDAUDCAP>"}'
    with open(caption_file_path, "r", encoding="utf-8") as f:
        captions = f.readlines()
        for caption in tqdm(captions, desc="Loading Captions"):
            #print(f"{type(caption)}")
            splits = caption.split(', "caption": ')
            caption = splits[1].replace('"}', '').replace("}", "")
            speaker_vid = splits[0].replace('{"fid": "', '').replace('"', '')
            vid = speaker_vid.rsplit("_", 1)[0]
            #print(f"vid: {vid} caption: {caption}")
            caption_dict[vid] = {"caption": caption.encode('utf-8').decode('unicode_escape'), "speaker_vid": speaker_vid}
        
    print(f"Total load {len(caption_dict)} valid captions.")
    return caption_dict


def list_files_recursive(directory, sufix=".json"):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(sufix):
                yield os.path.join(root, file).replace(directory+"/", "")


def get_file_list(data_dir, sufix=".json"):
    files = list(list_files_recursive(data_dir, sufix))
    return files


def get_asr(asr_path):
    asr = ""
    with open(asr_path, "r", encoding="utf-8") as f:
        asr_data = json.load(f)
    asr = asr_data["text"]
    return asr


def get_bbox(meta_path):
    if meta_path is None:
        return f"{0},{0},{1},{1}"  # left, top, right, down
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    bbox = meta_data["bbox"]
    bbox = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    return bbox
        

def build_meta(data_root, video_dir="videos", audio_dir="audios", asr_dir="asr", meta_path="", meta_dir=None, save_dir="./csv", num_data_each_batch=20000):
    """ Build metadata for dataset
    """
    os.makedirs(save_dir, exist_ok=True)

    # load captions
    captions = read_captions(meta_path)

    # get video paths
    video_paths = get_file_list(os.path.join(data_root, video_dir), sufix=".mp4")
    video_paths = {video_path.split('/')[-1].replace(".mp4", ""): osp.join(video_dir, video_path) for video_path in video_paths}
    # get audio paths
    audio_paths = get_file_list(os.path.join(data_root, audio_dir), sufix=".wav")
    audio_paths = {audio_path.split('/')[-1].replace(".wav", ""): osp.join(audio_dir, audio_path) for audio_path in audio_paths}
    # get asr paths
    asr_paths = get_file_list(os.path.join(data_root, asr_dir), sufix=".json")
    asr_paths = {asr_path.split('/')[-1].replace(".json", ""): osp.join(asr_dir, asr_path) for asr_path in asr_paths}
    # get meta paths
    meta_paths = None
    if meta_dir is not None:
        meta_paths = get_file_list(os.path.join(data_root, meta_dir), sufix=".json")
        meta_paths = {meta_path.split('/')[-1].replace(".json", ""): osp.join(meta_dir, meta_path) for meta_path in meta_paths}
    
    # collect all paths
    data_infos = []

    for vid in tqdm(asr_paths.keys(), desc="Collectting Meta Infos"):
        caption = captions[vid]["caption"]
        speaker_vid = captions[vid]["speaker_vid"]
        vid_name = vid.rsplit("_", 1)[0]
        video_path = video_paths.get(vid_name, None)
        audio_path = audio_paths.get(vid_name, None)
        if meta_paths is not None:
            meta_path = meta_paths.get(speaker_vid, None)
            if meta_path is None:
                continue
        else:
            meta_path = None
        if video_path is None:
           continue
        if audio_path is None:
           continue

        asr_path = asr_paths[vid]

        asr_path = osp.join(data_root, asr_path)
        meta_path = osp.join(data_root, meta_path) if meta_path is not None else None

        data_infos.append({
            "vid": vid,
            "video_path": video_path,
            "audio_path": audio_path,
            "bbox": get_bbox(meta_path),
            "asr": get_asr(asr_path),
            "caption": caption, 
        })

    print(f"Total has {len(data_infos)} valid samples.")
    
    # save meta files
    csv_path = save_dir + "/meta_%03d.csv"

    video_meta_df = pd.DataFrame(columns=list(data_infos[0].keys()))
    data_counter = 0
    file_index = 0
    for i, data in tqdm(enumerate(data_infos), desc="Saving Meta Files"):
        video_meta_df.loc[len(video_meta_df)] = data
        data_counter += 1
        if data_counter == num_data_each_batch:
            video_meta_df.to_csv(csv_path % file_index, index=False, encoding="utf-8")
            data_counter = 0
            file_index += 1
            video_meta_df = None
            del video_meta_df
            gc.collect()
            print(f"meta file {csv_path % file_index} has been saved.")
            video_meta_df = pd.DataFrame(columns=list(data_infos[0].keys()))

    if data_counter > 0:
        video_meta_df.to_csv(csv_path % file_index, index=False, encoding="utf-8")
        print(f"meta file {csv_path % file_index} has been saved.")    


if __name__ == "__main__":
    caption_file_path = "/Users/linqi/Documents/projects/reasearch/data/Edit/new_ovi.json"
    root_dir = "/Users/linqi/Documents/projects/reasearch/data/Edit/sample_1000_guaranteed2"
    #read_captions(caption_file_path=caption_file_path)
    build_meta(root_dir, meta_path=caption_file_path, save_dir="/Users/linqi/Documents/projects/reasearch/codes/data")