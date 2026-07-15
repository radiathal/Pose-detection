import tensorflow as tf
import numpy as np
import cv2

import time
import os

# Import matplotlib libraries   
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from threading import Thread
from threading import Event
from queue import Queue
import gesture_detect


target_fps = 4
stop_event = Event()
birth_t=0


# for video detect, define functions for each thread
def video_controller(image_q, file_name, birth_t):
  #open video and get framerate
  cap = cv2.VideoCapture(file_name)
  if not cap.isOpened():
        raise IOError(f"Cannot open video: {file_name}")
  video_fps = cap.get(cv2.CAP_PROP_FPS)
  if video_fps <= 0:
    raise ValueError("Could not read video FPS")

  frame_interval = video_fps / target_fps

  frame = None

  next_frame_to_save = 0.0
  frame_idx = 0
  i=0
  while not stop_event.is_set():
    try:
      if frame_idx >= next_frame_to_save:
        #print(f"[{time.time()-birth_t:.4f}]: {i} taking image...")
        ret, frame = cap.read()
        if not ret:
          break
        image_q.put(frame)
        next_frame_to_save += frame_interval
        #print(f"[{time.time()-birth_t:.4f}]: {i} taken image")
        i += 1
      frame_idx += 1
    except KeyboardInterrupt:
      break

def pose_detection_controller(image_q, display_q, birth_t):
  i = 0
  while not stop_event.is_set():
    try:
      image = image_q.get()
      print(f"[{time.time()-birth_t:.4f}]: {i} detecting image...")
      
      # Run model inference.
      keypoints_with_scores = gesture_detect.movenet(image)
      
      display_image = tf.expand_dims(image, axis=0)
      display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
      output_overlay = gesture_detect.draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

      display_q.put(output_overlay)
      print(f"[{time.time()-birth_t:.4f}]: {i} detected image")
      i+=1
    except KeyboardInterrupt:
      break


#starts other threads and handles display
def main_func(video_file_name):
  birth_t = time.time()
  print((f"[{time.time()-birth_t:.4f}]: START"))

  #make queues
  display_q = Queue()
  image_q = Queue()
  
  #init threads for different processes
  video_thread   = Thread(target=video_controller, args=(image_q, video_file_name, birth_t,))
  pose_thread    = Thread(target=pose_detection_controller, args=(image_q, display_q, birth_t,))
  #display_thread = Thread(target=display_controller, args=(display_q, img_display, fig,))
  
  #start threads
  video_thread.start()
  pose_thread.start()
  #display_thread.start()
  i=0
  while True:
    try:
      #run when you get processed image
      output_overlay = display_q.get()
      print(f"[{time.time()-birth_t:.4f}]: {i} displaying image...")
      cv2.imshow("Just Dance", output_overlay)
      print(f"[{time.time()-birth_t:.4f}]: {i} displayed image")
      cv2.waitKey(1)
      i+=1
    except KeyboardInterrupt:
      stop_event.set()
      cv2.destroyAllWindows()
      print((f"[{time.time()-birth_t:.4f}]: STOP"))
      break



if __name__ == "__main__":
  main_func('videos/cbto_follow.mp4')
