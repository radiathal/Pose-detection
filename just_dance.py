import tensorflow as tf
import numpy as np
import cv2

# audio 
import av
import sounddevice as sd

import time
import os

# Import matplotlib libraries   
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from threading import Thread, Event, Lock
from collections import deque
from queue import Queue, Empty
import gesture_detect


target_fps = 10
stop_event = Event()
start_event = Event()
birth_t=0



class Streamer():
  def audio_callback(self, outdata, frames, time_info, status):
    needed = len(outdata)
    buf = bytearray()
    with self.audio_lock:
      while len(buf) < needed and self.audio_buf:
        buf += self.audio_buf.popleft()
      if len(buf) < needed:
        buf += b'\x00' * (needed-len(buf)) #add silence if no data to add
        print("silence")
      elif len(buf) > needed:
        leftover = bytes(buf[needed:])
        self.audio_buf.appendleft(leftover)
        buf = buf[:needed]
        print("over")
    outdata[:] = bytes(buf)

  def __init__(self, video_file_name):
    self.container = av.open(video_file_name)
    self.video_stream = self.container.streams.video[0]
    self.audio_stream = self.container.streams.audio[0]

    self.sample_rate = self.audio_stream.rate
    self.channels    = self.audio_stream.channels

    self.resampler = av.AudioResampler(format='s16', layout=self.audio_stream.layout, rate=self.sample_rate)

    self.audio_lock = Lock()
    self.audio_buf = deque()


    #init audio stream
    self.stream = sd.RawOutputStream(
      samplerate=self.sample_rate,
      channels=self.channels,
      dtype='int16',
      callback=self.audio_callback,
      blocksize=2048,
      latency="high",
    )

  def close(self):
    self.stream.stop()
    self.stream.close()
    self.container.close()
    


    
  

# for video detect, define functions for each thread
def video_controller(image_q, streamer, birth_t):
  #only save select video frames according to target fps
  video_fps = float(streamer.video_stream.average_rate)
  frame_interval = video_fps / target_fps
  print(frame_interval)
  frame = None
  next_frame_to_save = 0.0
  frame_idx = 0
  i=0


  for packet in streamer.container.demux(streamer.video_stream, streamer.audio_stream):
    try:

      if packet.stream.type == "audio":
        for frame in packet.decode():
          for resampled in streamer.resampler.resample(frame):
            #data = bytes(resampled.planes[0])
            n_bytes = resampled.samples * len(resampled.layout.channels) * 2  # 2 = bytes per sample for s16
            data = bytes(resampled.planes[0])[:n_bytes]
            with streamer.audio_lock:
              streamer.audio_buf.append(data)

      elif packet.stream.type == "video":
        for frame in packet.decode():
          if frame_idx >= next_frame_to_save:
            #print(f"[{time.time()-birth_t:.4f}]: {i} unpacking frame...")
            img = frame.to_ndarray(format="bgr24") 
            ts = float(frame.pts * streamer.video_stream.time_base)
            image_q.put((ts, img))

            next_frame_to_save += frame_interval
            #print(f"[{time.time()-birth_t:.4f}]: {i} unpacked frame.")
            i += 1
          frame_idx += 1

      if stop_event.is_set():
        break
    except KeyboardInterrupt:
      break
      
  #signals that it is the end
  image_q.put(("STOP", "STOP"))
  streamer.container.close()
  


def camera_controller(image_q, birth_t):
  cap = cv2.VideoCapture(0)
  interval = 1.0/target_fps
  next_t = time.perf_counter()
  while not start_event.is_set():
    time.sleep(0.0001)
  while cap.isOpened() and not stop_event.is_set():  
    s = time.perf_counter()
    cap.grab()
    #print(f"GRAB TIME = {time.perf_counter()-s:.4f}")
    now_t = time.perf_counter()
    if now_t >= next_t:
      while now_t >= next_t:
        next_t += interval
      #print(f"[{time.time()-birth_t:.4f}]:  taking image...")
      ret, frame = cap.retrieve()
      if not ret:
        print("Error: Could not read frame.")
        break
      image_q.put(frame)
     # print(f"[{time.time()-birth_t:.4f}]:  taken image") 
  image_q.put("STOP")
  cap.release()


def pose_detection_controller(image_q, display_q, pose_prof, detector_index, birth_t):
  i = 0
  while not start_event.is_set():
    time.sleep(0.0001)
  while not stop_event.is_set():
    try:
      if detector_index == 0:
        data = image_q.get()
        ts = data[0]
        image = data[1]
      else:
        image = image_q.get()

      #if it's the end of the video, send STOP
      if type(image) == str:
        display_q.put(("STOP", "STOP"))
        break

      print(f"[{time.time()-birth_t:.4f}]: {i}.{detector_index} detecting image...")
      
      # Run model inference. (detector index for different gesture detection instances)
      landmarks = gesture_detect.detect_pose(image,i,detector_index) 
      if landmarks:
        pose_prof.update(landmarks)

      output_overlay = gesture_detect.draw_landmarks(image, landmarks)

      print(f"[{time.time()-birth_t:.4f}]: {i}.{detector_index} detected image")

      if detector_index == 0:
        display_q.put((ts, output_overlay))
      else:
        display_q.put(output_overlay)

      #mark image as processed
      image_q.task_done()
      
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
  audio_q  = Queue()

  vpose_prof = gesture_detect.PoseProfile()
  cpose_prof = gesture_detect.PoseProfile()
  
  # for audio unpacking from file
  streamer = Streamer(video_file_name)
 
  print("Loading...")
  
  #init threads for different processes
  camera_thread  = Thread(target=camera_controller, args=(cimage_q,birth_t,)) 
  vpose_thread   = Thread(target=pose_detection_controller, args=(vimage_q, vdisplay_q, vpose_prof, 0, birth_t,))
  cpose_thread   = Thread(target=pose_detection_controller, args=(cimage_q, cdisplay_q, cpose_prof, 1, birth_t,))
  video_thread   = Thread(target=video_controller, args=(vimage_q, streamer, birth_t,))
  
  #start threads
  video_thread.start()
  camera_thread.start()
  vpose_thread.start()
  cpose_thread.start()


  print("Ready!")

  i=0
  lag=0
  #last_t = time.perf_counter()
  start_time = time.perf_counter()
  start_event.set()
  
  while not stop_event.is_set():
    try:
      #wait for processed image

      ts, voutput_overlay = vdisplay_q.get()
      print(f"[{time.perf_counter()-start_time:.4f}]: {ts} got voutput")
      coutput_overlay     = cdisplay_q.get()
      print(f"[{time.perf_counter()-start_time:.4f}]: {ts} got coutput")

      if not streamer.stream.active:
        start_time = time.perf_counter()
        streamer.stream.start()
      
      
      if type(voutput_overlay) == str:
        stop_event.set()
        break
      print(f"[{time.perf_counter()-start_time:.4f}]: {ts} got image.")

      #if display is late skip the frame
      if not time.perf_counter() - start_time - ts > 1/target_fps: 
        #wait until timestamp
        while time.perf_counter() - start_time < ts - lag:
          time.sleep(0.001)

        print(f"[{time.perf_counter()-start_time:.4f}]: {ts} displaying image...")
        lag_s = time.perf_counter()
      
        #display
        cv2.imshow("Just Dance", voutput_overlay)

        cv2.imshow("Camera feed", coutput_overlay)
        print(f"[{time.perf_counter()-start_time:.4f}]: {ts} displayed c image")
        cv2.waitKey(1)
        lag = time.perf_counter() - lag_s

        #mark displays as processed
        cdisplay_q.task_done()
        vdisplay_q.task_done()
      i+=1

    except KeyboardInterrupt:
      stop_event.set()
      time.sleep(0.05)
      streamer.close()
      start_event.clear()
      cv2.destroyAllWindows()
      print((f"[{time.time()-birth_t:.4f}]: STOP"))
      break
  camera_thread.join()
  print("cam joined")
  video_thread.join()
  print("video joined")
  vpose_thread.join()
  print("vpose_joined")
  cpose_thread.join()
  print("cpose joined")
  



if __name__ == "__main__":
  main_func('videos/dsmn.mp4')
