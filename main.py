import argparse
import cv2
import onnxruntime
import numpy as np
import time

import box_utils


def preprocess_ultraface(image: cv2) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)


ONNX_DATA = {
    # modelname: filename, preprocess_func, (height, width)
    "ultraface_640": ("version-RFB-640.onnx", preprocess_ultraface, (640, 480)),
}

RED = (0, 255, 255)
WHITE = (255, 255, 255)


def main():
    parser = argparse.ArgumentParser("Test for ML Application")
    parser.add_argument("--print_time", default=False, type=bool)
    parser.add_argument("--model", default="ultraface_640", type=str)

    args = parser.parse_args()

    filename, func, size = ONNX_DATA[args.model]

    # VideoCapture オブジェクトを取得します
    capture = cv2.VideoCapture(0)

    sess = onnxruntime.InferenceSession(filename)

    while(True):
        times = []
        begin = time.time()
        times.append(begin)
        ret, frame = capture.read()
        times.append(time.time())
        # resize the window
        frame = cv2.resize(frame, size)
        img_arr = func(frame)
        times.append(time.time())
        rdict = sess.run(["scores", "boxes"], {"input": img_arr})
        times.append(time.time())

        # boxes (k, 4): an array of boxes kept
        # labels (k): an array of labels for each boxes kept
        # probs (k): an array of probabilities for each boxes being in corresponding labels
        boxes, labels, probs = box_utils.predict(
            size[0], size[1], rdict[0], rdict[1], 0.7)
        boxes = boxes[probs > 0.7]
        labels = labels[probs > 0.7]

        for box, label in zip(boxes, labels):
            pt0, pt1 = (box[0], box[1]), (box[2], box[3])
            frame = cv2.rectangle(frame, pt0, pt1, RED, thickness=1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(label), pt1, font, 4,
                        WHITE, 2, cv2.LINE_AA)

        cv2.imshow('title', frame)
        end = time.time()
        times.append(time.time())

        times = np.array(times)
        times = times[1:] - times[:len(times) - 1]
        times *= 1000.0

        if args.print_time:
            print(times)
            print(f"FPS:{(1.0 / (end - begin))}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
