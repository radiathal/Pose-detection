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


target_fps = 25
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
  print(video_fps)
  frame_interval = video_fps / target_fps
  print(frame_interval)

  frame = None

  next_frame_to_save = 0.0
  frame_idx = 0
  i=0
  while not stop_event.is_set():
    try:
      ret = cap.grab()
      if not ret:
        break
      if frame_idx >= next_frame_to_save:
        #print(f"[{time.time()-birth_t:.4f}]: {i} taking image...")
        ret, frame = cap.retrieve()
        if not ret:
          image_q.put("STOP")
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
  pose_prof = gesture_detect.PoseProfile()

  while not stop_event.is_set():
    try:
      image = image_q.get()
      if type(image) == str:
        break
      #print(f"[{time.time()-birth_t:.4f}]: {i} detecting image...")
      
      # Run model inference.
      landmarks = gesture_detect.detect_pose(image,i)
      if landmarks:
        pose_prof.update(landmarks)
        #pose_prof.print_profile()

      output_overlay = gesture_detect.draw_landmarks(image, landmarks)

      display_q.put(output_overlay)
      #print(f"[{time.time()-birth_t:.4f}]: {i} detected image")
      i+=1
    except KeyboardInterrupt:
      break


#starts other threads and handles display
def main_func(video_file_name):
  display_delay = 0.000001

  birth_t = time.time()
  print((f"[{time.time()-birth_t:.4f}]: START"))

  #make queues
  display_q = Queue()
  image_q = Queue()
  
  #init threads for different processes
  video_thread   = Thread(target=video_controller, args=(image_q, video_file_name, birth_t,))
  pose_thread    = Thread(target=pose_detection_controller, args=(image_q, display_q, birth_t,))
  
  #start threads
  video_thread.start()
  pose_thread.start()

  i=0
  last_t = time.perf_counter()
  while True:
    try:
      #wait for processed image
      output_overlay = display_q.get()
      #print(f"[{time.time()-birth_t:.4f}]: {i} got image.")

      #run display at correct framerate
      while time.perf_counter()-last_t < 1/target_fps:
        delay_time = (1/target_fps)-(time.perf_counter()-last_t)-display_delay
        if delay_time < 0:
          delay_time = 0
        time.sleep(delay_time)   #sleep for frame interval - elapsed time from last frame - displaying delay

      #print(f"[{time.time()-birth_t:.4f}]: {i} displaying image...")
      start_disp = time.perf_counter()

      #display
      cv2.imshow("Just Dance", output_overlay)

      #print(f"[{time.time()-birth_t:.4f}]: {i} displayed image")
      cv2.waitKey(1)
      i+=1

      #some timing stuff...
      display_delay = time.perf_counter() - start_disp  # how long it took to display image
      last_t = time.perf_counter()

    except KeyboardInterrupt:
      stop_event.set()
      cv2.destroyAllWindows()
      print((f"[{time.time()-birth_t:.4f}]: STOP"))
      break



if __name__ == "__main__":
  main_func('videos/test_video1_quality.mp4')
