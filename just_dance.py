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
          break
        image_q.put(frame)
        next_frame_to_save += frame_interval
        #print(f"[{time.time()-birth_t:.4f}]: {i} taken image")
        i += 1
      frame_idx += 1
    except KeyboardInterrupt:
      break

def camera_controller(image_q, birth_t):
  cap = cv2.VideoCapture(0)
  interval = 1.0/target_fps
  
  next_t = time.perf_counter()

  while cap.isOpened() and not stop_event.is_set():
    cap.grab()
    now_t = time.perf_counter()
    if now_t >= next_t:
      #print(f"[{time.time()-birth_t:.4f}]:  taking image...")
      ret, frame = cap.read()
      if not ret:
        print("Error: Could not read frame.")
        break
      image_q.put(frame)
      next_t += interval
      #print(f"[{time.time()-birth_t:.4f}]:  taken image") 


    

  cap.release()

def pose_detection_controller(vimage_q, cimage_q, display_q, birth_t):
  i = 0
  vpose_prof = gesture_detect.PoseProfile()
  cpose_prof = gesture_detect.PoseProfile()

  while not stop_event.is_set():
    try:
      vimage = vimage_q.get()
      cimage = cimage_q.get()
      #print(f"[{time.time()-birth_t:.4f}]: {i} detecting image...")
      
      # Run model inference. use 0 for video and 1 for camera detection (different gesture detection instances)
      landmarks = gesture_detect.detect_pose(vimage,i,0) 
      if landmarks:
        vpose_prof.update(landmarks)
      voutput_overlay = gesture_detect.draw_landmarks(vimage, landmarks)

      landmarks = gesture_detect.detect_pose(cimage,i,1) 
      if landmarks:
        vpose_prof.update(landmarks)
      coutput_overlay = gesture_detect.draw_landmarks(cimage, landmarks)

      display_q.put([voutput_overlay, coutput_overlay])
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
  vimage_q = Queue()
  cimage_q = Queue()
  
  #init threads for different processes
  video_thread   = Thread(target=video_controller, args=(vimage_q, video_file_name, birth_t,))
  camera_thread  = Thread(target=camera_controller, args=(cimage_q,birth_t,)) 
  pose_thread    = Thread(target=pose_detection_controller, args=(vimage_q, cimage_q, display_q, birth_t,))
  
  #start threads
  video_thread.start()
  camera_thread.start()
  pose_thread.start()

  i=0
  last_t = time.perf_counter()
  while True:
    try:
      #wait for processed image
      [voutput_overlay, coutput_overlay] = display_q.get()
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
      cv2.imshow("Just Dance", voutput_overlay)
      cv2.imshow("Camera feed", coutput_overlay)

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
  camera_thread.join()
  video_thread.join()
  pose_thread.join()



if __name__ == "__main__":
  main_func('videos/test_video1_quality.mp4')
