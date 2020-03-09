import cv2
import tensorflow as tf
import datetime
import argparse
import  sys
import os
import numpy as np



sys.path.insert(0,'./handtracking')
from utils import detector_utils as detector_utils


sys.path.insert(0,'./handrecog/src')
from predict import Predict_hand



detection_graph, sess = detector_utils.load_inference_graph()

def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np, IDs, probs):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            if(IDs[i] == 0):
                cv2.rectangle(image_np, p1, p2, (0, 255, 0), 3, 1)
                cv2.putText(image_np,
                            "Pers1" + ' ' + '%.04f'%probs[i],
                            (p1[0], p1[1] - 13),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1e-2 * (p2[1]-p1[1]) / 4,
                            (0, 255, 0), 2)
            if(IDs[i] == 1):
                cv2.rectangle(image_np, p1, p2, (255, 0, 0), 3, 1)
                cv2.putText(image_np,
                            "Pers1" + ' ' + '%.04f'%probs[i],
                            (p1[0], p1[1] - 13),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1e-2 * (p2[1]-p1[1]) / 4,
                            (255, 0, 0), 2)

# crop the detected bounding boxes on the images
def crop_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np, image_size):
    imgs = []
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            # p1 = (int(left), int(top))
            # p2 = (int(right), int(bottom))
            img = image_np[int(top):int(bottom),int(left):int(right)]
            img_resize = cv2.resize(img,(image_size,image_size))
            imgs.append(img_resize)
    return imgs

# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

    return image_np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-tar',
        '--target',
        dest='video_target',
        default=0,
        help='Path for saving the detected hand images.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=0,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    parser.add_argument('--model_dir', type=str,
                        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file',
                        default='/data/zming/models/hand/20190220-173420/')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)

    args = parser.parse_args()

    predict_hand = Predict_hand(args)

    cap = cv2.VideoCapture(args.video_source)

    # cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 2371);

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    vidname = str.split(args.video_source,'/')[-1]
    vidname = str.split(vidname,'.')[0]
    fourcc = cv2.cv.CV_FOURCC('M','J','P','G') ### opencv 2.x
    fps = 25
    videoWriter = cv2.VideoWriter(os.path.join(args.video_target, vidname+'_detect.avi'), fourcc, fps, (1920,1080))

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    # cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)
    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        # # draw bounding boxes on frame
        # detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
        #                                  scores, boxes, im_width, im_height,
        #                                  image_np)

        # Calculate Frames per second (FPS)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))

            if (args.fps > 0):
                imgs_crop = crop_box_on_image(num_hands_detect, args.score_thresh,
                                  scores, boxes, im_width, im_height,
                                  image_np, args.image_size)
                if imgs_crop:
                    IDs, probs = predict_hand.recog_hand(np.array(imgs_crop))

                    draw_fps_on_image("FPS : " + str(int(fps)), image_np)

                    # draw bounding boxes on frame
                    draw_box_on_image(num_hands_detect, args.score_thresh,
                                                     scores, boxes, im_width, im_height,
                                                     image_np, IDs, probs)

                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.video_target,vidname, 'frame%4d.jpg'%num_frames),image_np)
                videoWriter.write(image_np)


        num_frames += 1

    videoWriter.release()