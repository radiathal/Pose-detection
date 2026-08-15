import gesture_detect
from queue     import Queue
from threading import Thread
import av


class Streamer():
  def __init__(self, video_file_name):
    self.container = av.open(video_file_name)
    self.video_stream = self.container.streams.video[0]
    self.audio_stream = self.container.streams.audio[0]

    self.sample_rate = self.audio_stream.rate
    self.channels    = self.audio_stream.channels

    self.resampler = av.AudioResampler(format='s16', layout=self.audio_stream.layout, rate=self.sample_rate)


  def close(self):
    self.container.close()

def create_jd_folder():
  Pass


def video_process(streamer, target_fps, frame_q):
  #only save select video frames according to target fps
  video_fps = float(streamer.video_stream.average_rate)
  frame_interval = video_fps / target_fps

  #init
  frame = None
  next_frame_to_save = 0.0
  frame_idx = 0


  for packet in streamer.container.demux(streamer.video_stream, streamer.audio_stream):
    if packet.stream.type == "audio":
      for frame in packet.decode():
        for resampled in streamer.resampler.resample(frame):
          n_bytes = resampled.samples * len(resampled.layout.channels) * 2  # 2 = bytes per sample for s16
          data = bytes(resampled.planes[0])[:n_bytes]
          
          #PUT DATA SOMEWHERE

    elif packet.stream.type == "video":
      for frame in packet.decode():
        if frame_idx >= next_frame_to_save:
          img = frame.to_ndarray(format="bgr24") 
          ts = float(frame.pts * streamer.video_stream.time_base)
          frame_q.put([img, ts])

          next_frame_to_save += frame_interval
        frame_idx += 1

  frame_q.put("STOP")
  streamer.container.close()

def frame_pose_detect(pose_prof, frame_q): 
  i=0
  while True:
    data = frame_q.get()
    if type(data) == str:
      break
    image = data[0]
    ts = data[1]

    # Run model inference. (detector index for different gesture detection instances)
    landmarks = gesture_detect.detect_pose(image,i,detector_index) 
    output_overlay = gesture_detect.draw_landmarks(image, landmarks)
    # PUT DATA SOMEWHERE

    pose_prof.update(landmarks)
    #PUT DATA SOMEWHERE
      
    i+=1



if __name__ == "__main__":
  target_fps = input("target fps: ")
  video      = input("video filename: ")
  streamer = Streamer(video) # MAKE EXCEPTION FOR ERROR

  pose_prof = gesture_detect.PoseProfile()


  frame_q = Queue()
  unpack_thread      = Thread(target=video_process, args=(streamer,target_fps, frame_q,))
  pose_detect_thread = Thread(target=frame_pose_detect, args=(pose_prof, frame_q,))
