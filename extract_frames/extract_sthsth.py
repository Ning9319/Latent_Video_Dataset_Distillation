import os
import os.path as osp
import json
import pdb
from functools import cached_property, lru_cache
from PIL import Image
import tqdm
import multiprocessing
import time


# change the following paths 
ROOT = "./sthv2"
OUT_DIR = "./SSv2"
N_FRAMES = 16
RE_SIZE = 112


with open(osp.join(ROOT, "something-something-v2-labels.json")) as fp:
    verb2index_dict = json.load(fp)
    verb2index_dict = {k.lower():int(v) for k, v in verb2index_dict.items()}
    rev_verb_list = {v: k for k, v in verb2index_dict.items()}
    verb_list = [rev_verb_list[i] for i in range(len(rev_verb_list))]


sthsth_annot_path = {
    "train": "something-something-v2-train.json",
    "val": "something-something-v2-validation.json",
}

@lru_cache(maxsize=None)
def frame_number(data_root: str):
    return json.load(open(osp.join(data_root, "frame_number.json")))

def process_image(in_path, out_path):
    original_image = Image.open(in_path)
    resized_image = original_image.resize((RE_SIZE, RE_SIZE))
    resized_image.save(out_path) 


def mulp_process(args):
    video_id, total_frames = args

    image_name_template = osp.join("rawframes", video_id, "img_{:05d}.jpg")

    folder = osp.join(OUT_DIR, "rawframes", video_id)
    if os.path.exists(folder):
        return
    else:
        os.makedirs(folder)

    if total_frames < N_FRAMES:
        selected_ids = [i + 1 for i in range(total_frames)]  # Move everything by +1
        while len(selected_ids) < N_FRAMES:
            selected_ids.append(selected_ids[-1])  # Duplicate the last frame until 16 are reached
  
    else:
        max_fps = (total_frames-1)//(N_FRAMES-1)
        remain = (total_frames - 1) % (N_FRAMES - 1)
        selected_ids = [i + 1 for i in range(remain // 2, total_frames, max_fps)][:N_FRAMES]  # Shift by +1
    
    assert len(selected_ids) == N_FRAMES, f"Error: {selected_ids} (max_fps: {max_fps}, total_frames: {total_frames})"

    for idx in selected_ids:
        fname = image_name_template.format(idx)
        process_image(
            osp.join(ROOT, fname),
            osp.join(OUT_DIR, fname)
        )


if __name__ == "__main__":  
    json.dump(verb_list, open(osp.join(OUT_DIR, f"class_list.json"), "w"))

    for split, path in sthsth_annot_path.items():
        print(split)
        with open(osp.join(ROOT, path)) as fp:
            sthsth_raw_annot = json.load(fp)

        sthsth_annot = []
        mulp_tasks = []
        for sample in sthsth_raw_annot:
            video_id = sample['id']
            verb_str = sample['template'].replace("[", "").replace("]", "").lower()
            verb_id  = verb2index_dict[verb_str]

            sthsth_annot.append({
                "id": video_id,
                "class": verb_id,
            })

            total_frames = frame_number(ROOT)[video_id]
            mulp_tasks.append([video_id, total_frames])

        json.dump(sthsth_annot, open(osp.join(OUT_DIR, f"annot_{split}.json"), "w"))

        # Multiprocessing Fix: Wrap it inside __main__
        with multiprocessing.Pool(32) as mulpool:
            for _ in tqdm.tqdm(mulpool.imap(mulp_process, mulp_tasks), total=len(mulp_tasks), ncols=60):
                pass

    pdb.set_trace()  # Debugging