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
from queue import Queue, Empty, Full
import gesture_detect

#flip camera image vertically
from PIL import Image


target_fps = 10    #10 is reliable
memory_seconds = 3 #how many seconds of future frames are stored in memory 
stop_event = Event()
start_event = Event()
birth_t=0

#rating
DRAW_RATING_LIGHT_MODE = 0
l_accuracy_display_location = [(50,50), (50,50)]
l_accuracy_display_width = 50

DRAW_RATING_BAR_MODE = 1
b_accuracy_display_location = [(0,0), (0,80)]
b_accuracy_display_width = 0


rating_mode = gesture_detect.POINT_MODE
draw_rating_mode = DRAW_RATING_BAR_MODE
draw_rating_rate = 1 * target_fps #save rating every x frames


#just dance screen too big or small? change value.
resize_window = 0.5





class Streamer():
  def audio_callback(self, outdata, frames, time_info, status):
    needed = len(outdata)
    buf = bytearray()
    with self.audio_lock:
      while len(buf) < needed and self.audio_buf:
        buf += self.audio_buf.popleft()
      if len(buf) < needed:
        buf += b'\x00' * (needed-len(buf)) #add silence if no data to add
        print("RAN OUT OF AUDIO DATA!")
      elif len(buf) > needed:
        leftover = bytes(buf[needed:])
        self.audio_buf.appendleft(leftover)
        buf = buf[:needed]
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
      latency="low",
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

            #try putting frame into queue. If code is terminated, don't get stuck on q.put()
            try:
              image_q.put((ts, img), timeout=2)
            except Full:
              continue

            next_frame_to_save += frame_interval
            #print(f"[{time.time()-birth_t:.4f}]: {i} unpacked frame.")
            i += 1
          frame_idx += 1

      if stop_event.is_set():
        break
    except KeyboardInterrupt:
      break
      
  #signals that it is the end
  try:
    image_q.put(("STOP", "STOP"), timeout=2)
  except Full:
    pass
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
  try:
    image_q.put("STOP", timeout=2)
  except Full:
    pass
  cap.release()


def pose_detection_controller(image_q, display_q, rating_q, detector_index, birth_t):
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
        if type(image) != str:
          image = cv2.flip(image, 1) #flip image from camera horisontally

      #if it's the end of the video, send STOP
      if stop_event.is_set():
        break
      if type(image) == str:
        try:
          display_q.put(("STOP", "STOP"), timeout=2)
        except Full:
          pass
        try:
          rating_q.put(None, timeout=2)
        except Full:
          pass
        break

      #print(f"[{time.time()-birth_t:.4f}]: {i}.{detector_index} detecting image...")
      
      # Run model inference. (detector index for different gesture detection instances)
      landmarks = gesture_detect.detect_pose(image,i,detector_index) 
      if landmarks:
        rating_q.put(landmarks)
      else: 
        rating_q.put(None)

      output_overlay = gesture_detect.draw_landmarks(image, landmarks)

      #print(f"[{time.time()-birth_t:.4f}]: {i}.{detector_index} detected image")

      if detector_index == 0:
        try:
          display_q.put((ts, output_overlay), timeout=2)
        except Full:
          print("Display_queue is Full!!")
      else:
        try:
          display_q.put(output_overlay, timeout=2)
        except Full:
          print("Display_queue is Full!!")

      #mark image as processed
      image_q.task_done()
      
      i+=1
    except KeyboardInterrupt:
      break

def rating_controller(vrating_q, crating_q, accuracy_q, birth_t):
  #show accuracy one frame later
  accuracy_q.put(0)

  vpose_prof = gesture_detect.PoseProfile()
  cpose_prof = gesture_detect.PoseProfile()
  i = 0
  while not start_event.is_set():
    time.sleep(0.0001)
  while not stop_event.is_set():
    try:
      try:
        vlandmarks = vrating_q.get(timeout=2)
      except Empty:
        continue
      try:
        clandmarks = crating_q.get(timeout=2)
      except Empty:
        continue

      if vlandmarks != None and clandmarks != None:
        #print((f"[{time.time()-birth_t:.4f}]: {i} accs..."))
        vpose_prof.update(vlandmarks)
        cpose_prof.update(clandmarks)

        accuracies = gesture_detect.compare_pose_profiles(vpose_prof, cpose_prof, rating_mode)

        accuracy_q.put(accuracies)
        #print((f"[{time.time()-birth_t:.4f}]: {i} accs done"))
      else:
        print("----------------No LANDMARKS!!!------------------")
        accuracy_q.put(None)
      i += 1

    except KeyboardInterrupt:
      break

#for testing only (?)
def draw_accuracy(image, accuracies, video_length, mode, ongoing = None):
  
  if mode == DRAW_RATING_LIGHT_MODE:
    if accuracies != None:
      i=0
      #list
      if type(accuracies) != int:
        for acc in accuracies:
          #greaner the closer accuracy is to 1
          color = (0, int(acc*255), int((1-acc)*255))
          cv2.line(image, (l_accuracy_display_location[0][0], l_accuracy_display_location[0][1]+l_accuracy_display_width*i), 
                          (l_accuracy_display_location[1][0], l_accuracy_display_location[1][1]+l_accuracy_display_width*i), 
                          color, 
                        l_accuracy_display_width)
          i+=1
      #int
      else: 
        color = (0, int(accuracies*255), int((1-accuracies)*255))
        cv2.line(image, l_accuracy_display_location[0], 
                        l_accuracy_display_location[1], 
                        color, 
                        l_accuracy_display_width)
  
  elif mode == DRAW_RATING_BAR_MODE:
    if ongoing:
      b_accuracy_display_width = ongoing[0]
      i = ongoing[1]
      gained = ongoing[2]
      all_accuracies = ongoing[3]
      #print(all_accuracies)

    if b_accuracy_display_width == 0:
      b_accuracy_display_width = int((draw_rating_rate/video_length) * image.shape[1])
      #print(f"{draw_rating_rate}/{video_length} * {image.shape[1]} = {b_accuracy_display_width}")

    #draw back as black
    color = (0, 0, 0)
    cv2.line(image, (0,              int(b_accuracy_display_location[1][1]/2)), 
                    (image.shape[1], int(b_accuracy_display_location[1][1]/2)), 
                    color, 
                    b_accuracy_display_location[1][1])


    if i < draw_rating_rate:
      if accuracies == None and all_accuracies != []:
        accuracies = all_accuracies[-1]
      elif type(accuracies) != int and accuracies != None:
        accuracies = accuracies[0]
      if accuracies != None:
        gained += accuracies
        #print(f"GAINED += {accuracies}")
        i += 1
    if i == draw_rating_rate:
      all_accuracies.append(gained/i)
      #print(f"ACCURACY = {gained}/{i} = {gained/(i)}")
      gained = 0
      i=0

    j=0
    for acc in all_accuracies:
      #greaner the closer accuracy is to 1
      color = (0, int(acc*255), int((1-acc)*255))
      cv2.line(image, (b_accuracy_display_location[0][0]+b_accuracy_display_width*j, b_accuracy_display_location[0][1]), 
                      (b_accuracy_display_location[1][0]+b_accuracy_display_width*j, b_accuracy_display_location[1][1]), 
                      color, 
                      b_accuracy_display_width)
      j+=1
    ongoing = [b_accuracy_display_width, i, gained, all_accuracies]

    

  return image, ongoing
  



#starts other threads and handles display
def main_func(video_file_name):
  display_delay = 0.000001

  print(("[00.00]: START"))

  saved_frames = target_fps * memory_seconds 

  #make queues
  vdisplay_q = Queue(maxsize=saved_frames)
  cdisplay_q = Queue(maxsize=saved_frames)
  vimage_q = Queue(maxsize=saved_frames)
  cimage_q = Queue(maxsize=saved_frames)
  audio_q  = Queue()
  vrating_q = Queue()
  crating_q = Queue()
  accuracy_q = Queue()

  
  
  # for audio unpacking from file
  streamer = Streamer(video_file_name)
  video_length = float(streamer.video_stream.duration * streamer.video_stream.time_base * target_fps)

  
  #init threads for different processes
  camera_thread  = Thread(target=camera_controller, args=(cimage_q,birth_t,)) 
  vpose_thread   = Thread(target=pose_detection_controller, args=(vimage_q, vdisplay_q, vrating_q, 0, birth_t,))
  cpose_thread   = Thread(target=pose_detection_controller, args=(cimage_q, cdisplay_q, crating_q, 1, birth_t,))
  video_thread   = Thread(target=video_controller, args=(vimage_q, streamer, birth_t,))
  rating_thread  = Thread(target=rating_controller, args=(vrating_q,crating_q, accuracy_q, birth_t))
  
  #start threads
  video_thread.start()
  camera_thread.start()
  vpose_thread.start()
  cpose_thread.start()
  rating_thread.start()


  i=0
  lag=0
  ongoing = [b_accuracy_display_width, 0, 0, []]
  #last_t = time.perf_counter()
  start_time = time.perf_counter() + streamer.stream.latency
  start_event.set()
  
  while not stop_event.is_set():
    try:
      #wait for processed image
      accuracies = accuracy_q.get()
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
      print(f"[{time.perf_counter()-start_time:.4f}]: {ts} got images.")

      #if display is late skip the frame
      if not time.perf_counter() - start_time - ts > 1/target_fps: 
        #wait until timestamp
        while time.perf_counter() - start_time < ts - lag:
          time.sleep(0.001)

        print(f"[{time.perf_counter()-start_time:.4f}]: {ts} displaying image...")
        lag_s = time.perf_counter()
      
        #display
        resized = cv2.resize(voutput_overlay, None, fx = resize_window, fy = resize_window, interpolation = cv2.INTER_AREA)
        resized, ongoing = draw_accuracy(resized, accuracies, video_length, draw_rating_mode, ongoing)
        cv2.imshow("Just Dance", resized)

        cv2.imshow("Camera feed", coutput_overlay)
        print(f"[{time.perf_counter()-start_time:.4f}]: {ts} displayed images")
        cv2.waitKey(1)
        lag = time.perf_counter() - lag_s

        #mark displays as processed
        cdisplay_q.task_done()
        vdisplay_q.task_done()
        accuracy_q.task_done()
      i+=1

    except KeyboardInterrupt:
      stop_event.set()
      time.sleep(0.05)
      streamer.close()
      start_event.clear()
      cv2.destroyAllWindows()
      print((f"[{time.perf_counter()-start_time:.4f}]: STOP"))
      break
  camera_thread.join()
  print("cam joined")
  video_thread.join()
  print("video joined")
  vpose_thread.join()
  print("vpose_joined")
  cpose_thread.join()
  print("cpose joined")
  rating_thread.join()
  print("rating joined")
  



if __name__ == "__main__":
  main_func('videos/dsmn.mp4')
