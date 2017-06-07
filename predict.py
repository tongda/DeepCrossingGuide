import argparse
import math
import time
from pathlib import Path

import numpy as np
from moviepy.editor import VideoFileClip

import cv2
from crossing_guide import CrossingGuide


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str,
                        help="Model path.")
    parser.add_argument("--all-feat", dest="all_feat", action="store_true",
                        help="Whether to use all features.")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="test_output",
                        help="Directory of output images.")
    return parser


def draw_arrow(image, p, q, color, arrowMagnitude=9, thickness=1, line_type=8, shift=0):
    # Draw the principle line
    output = np.copy(image)
    output = cv2.line(output, p, q, color, thickness, line_type, shift)
    # compute the angle alpha
    angle = math.atan2(p[1] - q[1], p[0] - q[0])
    # compute the coordinates of the first segment
    t = (int(q[0] + arrowMagnitude * math.cos(angle + math.pi / 4)),
         int(q[1] + arrowMagnitude * math.sin(angle + math.pi / 4)))
    # //Draw the first segment
    output = cv2.line(output, t, q, color, thickness, line_type, shift)
    # //compute the coordinates of the second segment
    t = (int(q[0] + arrowMagnitude * math.cos(angle - math.pi / 4)),
         int(q[1] + arrowMagnitude * math.sin(angle - math.pi / 4)))
    # //Draw the second segment
    output = cv2.line(output, t, q, color, thickness, line_type, shift)
    return output


def main(conf):
    root = Path("./data/0524")
    ts_range = (149548990624, 149548992008)
    guide = CrossingGuide(save_path=conf.model)
    guide.load()

    output_dir = Path(conf.output_dir)
    output_dir.mkdir(exist_ok=True)

    proc_times = []
    for f in root.glob("*.jpg"):
        if ts_range[1] > int(f.stem) > ts_range[0]:
            img = cv2.imread(str(f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cur_time = time.time()
            prediction = guide.predict(img)
            proc_times.append(time.time() - cur_time)
            output = cv2.putText(
                img,
                "{0[0]:.2}, {0[1]:.2}, {0[2]:.2}".format(prediction[0]),
                (10, 15), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=(0, 255, 0), thickness=2)
            if conf.all_feat:
                output = cv2.putText(
                    output,
                    "{0[3]:.2}, {0[4]:.2}, {0[5]:.2}".format(prediction[0]),
                    (10, 30), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=(0, 255, 0), thickness=2)
                output = cv2.putText(
                    output,
                    "{0[6]:.2}, {0[7]:.2}, {0[8]:.2}".format(prediction[0]),
                    (10, 45), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=(0, 255, 0), thickness=2)
                output = cv2.putText(
                    output,
                    "{0[9]:.2}, {0[10]:.2}, {0[11]:.2}".format(prediction[0]),
                    (10, 60), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=(0, 255, 0), thickness=2)
            mid = (img.shape[1] // 2, img.shape[0] // 2)
            arrow_length = 100
            target = (int(img.shape[1] // 2 - arrow_length * math.sin(prediction[0][1])), int(
                img.shape[0] // 2 - arrow_length * math.cos(prediction[0][1])))
            output = draw_arrow(output, mid, target, (255, 0, 0), 10, 2)
            cv2.imwrite(str(output_dir / "{}-predicted.jpg").format(f.stem),
                        cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print("Finished. Average processing time per image: {:.2f} ms".format(sum(proc_times) / len(proc_times) * 1000))


if __name__ == "__main__":
    parser = create_parser()
    conf = parser.parse_args()
    main(conf)
