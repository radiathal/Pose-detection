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


target_fps = 20
stop_event = Event()
start_event = Event()
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
  #signals that it is the end
  image_q.put("STOP")
  


def camera_controller(image_q, birth_t):
  cap = cv2.VideoCapture(0)
  interval = 1.0/target_fps
  next_t = time.perf_counter()

  while cap.isOpened() and not stop_event.is_set() and start_event.is_set():  
    s = time.perf_counter()
    cap.grab()
    print(f"GRAB TIME = {time.perf_counter()-s:.4f}")
    now_t = time.perf_counter()
    if now_t >= next_t:
      while next_t <= now_t:
        next_t += interval
      print(f"[{time.time()-birth_t:.4f}]:  taking image...")
      ret, frame = cap.retrieve()
      if not ret:
        print("Error: Could not read frame.")
        break
      image_q.put(frame)
      print(f"[{time.time()-birth_t:.4f}]:  taken image") 
  cap.release()


def pose_detection_controller(image_q, display_q, pose_prof, detector_index, birth_t):
  i = 0
  while not start_event.is_set():
    time.sleep(0.0001)
  while not stop_event.is_set():
    try:
      image = image_q.get()

      #if it's the end of the video, send STOP
      if type(image) == str:
        display_q.put("STOP")
        break

      print(f"[{time.time()-birth_t:.4f}]: {i}.{detector_index} detecting image...")
      
      # Run model inference. (detector index for different gesture detection instances)
      landmarks = gesture_detect.detect_pose(image,i,detector_index) 
      if landmarks:
        pose_prof.update(landmarks)

      #print(f"[{time.time()-birth_t:.4f}]: {i} drawing image...")
      output_overlay = gesture_detect.draw_landmarks(image, landmarks)
      #print(f"[{time.time()-birth_t:.4f}]: {i} drawn image")

      print(f"[{time.time()-birth_t:.4f}]: {i}.{detector_index} detected image")

      display_q.put(output_overlay)
      
      i+=1
    except KeyboardInterrupt:
      break


#starts other threads and handles display
def main_func(video_file_name):
  display_delay = 0.000001

  birth_t = time.time()
  print((f"[{time.time()-birth_t:.4f}]: START"))

  #make queues
  vdisplay_q = Queue()
  cdisplay_q = Queue()
  vimage_q = Queue()
  cimage_q = Queue()

  vpose_prof = gesture_detect.PoseProfile()
  cpose_prof = gesture_detect.PoseProfile()
  
  #process the just dance video before starting
  print("Loading...")
  video_thread   = Thread(target=video_controller, args=(vimage_q, video_file_name, birth_t,))
  video_thread.start()
  video_thread.join()
  print("Ready!")
  
  #init threads for different processes
  camera_thread  = Thread(target=camera_controller, args=(cimage_q,birth_t,)) 
  vpose_thread   = Thread(target=pose_detection_controller, args=(vimage_q, vdisplay_q, vpose_prof, 0, birth_t,))
  cpose_thread   = Thread(target=pose_detection_controller, args=(cimage_q, cdisplay_q, cpose_prof, 1, birth_t,))
  
  #start threads
  camera_thread.start()
  vpose_thread.start()
  cpose_thread.start()

  i=0
  last_t = time.perf_counter()
  start_event.set()
  while not stop_event.is_set():
    try:
      #wait for processed image
      voutput_overlay = vdisplay_q.get()
      coutput_overlay = cdisplay_q.get()
      if type(voutput_overlay) == str:
        stop_event.set()
        break
      #print(f"[{time.time()-birth_t:.4f}]: {i} got image.")

      #run display at correct framerate
      #while time.perf_counter()-last_t < 1/target_fps:
        #delay_time = (1/target_fps)-(time.perf_counter()-last_t)-display_delay
        #if delay_time < 0:
          #delay_time = 0
        #time.sleep(delay_time)   #sleep for frame interval - elapsed time from last frame - displaying delay

      print(f"[{time.time()-birth_t:.4f}]: {i} displaying image...")
      start_disp = time.perf_counter()

      #display
      cv2.imshow("Just Dance", voutput_overlay)
      print(f"[{time.time()-birth_t:.4f}]: {i} displayed v image")

      cv2.imshow("Camera feed", coutput_overlay)
      print(f"[{time.time()-birth_t:.4f}]: {i} displayed c image")
      cv2.waitKey(1)
      i+=1

      #some timing stuff...
      display_delay = time.perf_counter() - start_disp  # how long it took to display image
      last_t = time.perf_counter()

    except KeyboardInterrupt:
      stop_event.set()
      start_event.clear()
      cv2.destroyAllWindows()
      print((f"[{time.time()-birth_t:.4f}]: STOP"))
      break
  camera_thread.join()
  #video_thread.join()
  cpose_thread.join()
  vpose_thread.join()



if __name__ == "__main__":
  main_func('videos/test_video1_quality.mp4')
