import json
import random
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
import sys

sys.stderr.write("Start of file")


def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    # var = 0.1
    # sigma = var**0.5
    gauss = np.random.normal(mean, 1.5, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy


random.seed(42)
for split in ["train", "test"]:
    vid_path_inp = Path("dataset/data/videos/YT_4sec") / split / "input"
    video_names = [vid.stem for vid in vid_path_inp.glob("*")]
    random.shuffle(video_names)
    output_size = (int(2048 / 4), int(1080 / 4))
    lower_green = np.array([0, 125, 0])
    upper_green = np.array([100, 255, 120])
    MAX_DURATION = 4
    out_path = Path("data/images/YT_4sec") / split
    bgpath = Path("dataset/data/images/backgrounds") / split
    label_out_path = out_path / "labels"
    input_out_path = out_path / "input"
    label_out_path.mkdir(parents=True, exist_ok=True)
    input_out_path.mkdir(parents=True, exist_ok=True)
    frame_counter = 0
    count_lbl = 0
    out_log = defaultdict(list)
    for i, vid in enumerate(video_names):
        sys.stderr.write("Video: {}".format(vid))
        bgimg = [img for img in bgpath.glob("*")]
        bgimg = str(bgimg[i % len(bgimg)])
        bgimg = cv2.imread(bgimg)
        bgimg = cv2.resize(bgimg, output_size)
        # bgimg = np.clip(add_noise(bgimg), a_min=0, a_max=255)
        start = True
        print("--------------------------------")
        print("video: ", vid)
        new_vid_marker = True
        cap_inp = cv2.VideoCapture(str(vid_path_inp / vid) + ".mp4")
        while cap_inp.isOpened():
            ret, frame = cap_inp.read()
            out_name = (str(frame_counter).zfill(5) + ".jpg")
            if ret:
                mask = cv2.inRange(frame, lower_green,
                                   upper_green)  # create a mask for the label and the background swap
                mask = np.expand_dims(mask, axis=-1)
                label = np.where(mask, (0, 0, 0), (255, 255, 255))
                out_img = np.where(mask, bgimg, frame)
                out_name = str(frame_counter).zfill(5) + ".jpg"
                cv2.imwrite(str(input_out_path / out_name), np.uint8(out_img))
                cv2.imwrite(str(label_out_path / out_name), np.uint8(label))
                out_log["inputs"].append((str(input_out_path / out_name), int(new_vid_marker)))
                out_log["labels"].append((str(label_out_path / out_name), int(new_vid_marker)))
                frame_counter += 1
                new_vid_marker = False
            else:
                break
        cap_inp.release()

    with open(str("dataset" / out_path / "out_log.json"), "w") as js:
        json.dump(dict(out_log), js)
