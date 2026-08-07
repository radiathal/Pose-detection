import cv2
import numpy as np
from time import time
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt

#setting up pose detection...
BaseOptions = tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

model_path = "./pose_landmarker_full.task"  

# PoseLandmarker options
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detectors = [PoseLandmarker.create_from_options(options), PoseLandmarker.create_from_options(options)]



# Skeleton connections for drawing
POSE_CONNECTIONS = [
    (11,13),(13,15),(15,17),
    (12,14),(14,16),(16,18),
    (11,12),
    (11,23),(12,24),
    (23,24),
    (23,25),(25,27),
    (24,26),(26,28)
]

POINT_COL = (255,255,0)
CONNECTION_COL = (255,120,0)

#map out points and angles to names
point_dict = {
        "head"       : 0,
        "shoulder_l" : 11,
        "shoulder_r" : 12,
        "elbow_l" : 13,
        "elbow_r" : 14,
        "wrist_l" : 15,
        "wrist_r" : 16,
        "hand_l"  : 17,
        "hand_r"  : 18,
        "hip_l"   : 23,
        "hip_r"   : 24,
        "knee_l"  : 25,
        "knee_r"  : 26,
        "ankle_l" : 27,
        "ankle_r" : 28
    }
angle_dict = {
        "shoulder_l" : (13, 11, 12),
        "shoulder_r" : (14, 12, 11),
        "elbow_l" : (15, 13, 11),
        "elbow_r" : (16, 14, 12),
        "hip_l"   : (25, 23, 24),
        "hip_r"   : (26, 24, 23),
        "knee_l"  : (27, 25, 23),
        "knee_r"  : (28, 26, 24),
        #"wrist_l" : (17, 15, 13),
        #"wrist_r" : (18, 16, 14)
    }


###############################################

class PoseProfile:
  def __init__(self):
    self.points   = []
    self.l_points = []
    self.point_vels = []

    self.angles   = []
    self.l_angles = []
    self.angle_vels = []
    

  def update(self, landmarks):
    self.l_points = self.points.copy()
    self.points.clear()
    #get new points
    for point in point_dict.values():
      self.points.append([landmarks[point].x, landmarks[point].y])

    self.l_angles = self.angles.copy()
    self.angles.clear()
    #get new angles
    for angle in angle_dict.values():
      #coordinates of points
      a = np.array((landmarks[angle[0]].x, landmarks[angle[0]].y))
      b = np.array((landmarks[angle[1]].x, landmarks[angle[1]].y))
      c = np.array((landmarks[angle[2]].x, landmarks[angle[2]].y))
      # "b becomes origin"
      ba = a - b
      bc = c - b
      #get angles ba and bc
      angle_a = np.arctan2(ba[1], [ba[0]]) #takes in y, x
      angle_c = np.arctan2(bc[1], [bc[0]]) 
      self.angles.append(angle_c - angle_a)

    #get changes in angles and point positions
    try:
      self.angle_vels = np.array(self.angles) - np.array(self.l_angles)
      self.point_vels = np.array(self.points) - np.array(self.l_points)
    except ValueError: #means that l_angles/l_points havent been yet set up
      self.angle_vels = self.angles.copy()
      self.point_vels = self.points.copy() 

  def print_profile(self):
    i = 0
    print("POINTS:")
    for key in point_dict.keys():
      print(f"{key}: {self.points[i]}, speed: {self.point_vels[i]}")
      i += 1
    print("ANGLES:")
    i = 0
    for key in angle_dict.keys():
      print(f"{key}: {self.angles[i]}, speed: {self.angle_vels[i]}")
      i += 1

def compare_pose_profiles(p1, p2, accuracy):
  point_accuracy  = 0
  pointd_accuracy = 0
  angle_accuracy  = 0
  angled_accuracy = 0
  return point_accuracy, pointd_accuracy, angle_accuracy, angled_accuracy


#################################################


##FUNCTIONS FOR POSE DETECTION AND DISPLAYING DRAWN SKELETON
def detect_pose(image, frame_id, detector_ind = 0):
   # Convert to RGB for Mediapipe
   rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

   # detect
   results = detectors[detector_ind].detect_for_video(mp_image, frame_id)

   if results.pose_landmarks:
     landmarks = results.pose_landmarks[0]
   else:
     landmarks = None
   return landmarks

def draw_landmarks(image, landmarks):
  if landmarks == None:
    return image

  h, w, _ = image.shape

  for p in point_dict.values():
    lm = landmarks[p]
    x, y = int(lm.x * w), int(lm.y * h)
    cv2.circle(image, (x, y), 4, POINT_COL, -1)
    for p1, p2 in POSE_CONNECTIONS:
      x1, y1 = int(landmarks[p1].x * w), int(landmarks[p1].y * h)
      x2, y2 = int(landmarks[p2].x * w), int(landmarks[p2].y * h)
      cv2.line(image, (x1, y1), (x2, y2), CONNECTION_COL, 2)
  return image


  ########################################


  #TESTING
def camera_feed_detect():
  cap = cv2.VideoCapture(0)
  frame_id = 0

  while cap.isOpened():
    #take the image
    start_time = time()
    ret, frame = cap.read()

    if not ret:
      print("Error: Could not read frame.")
      continue
    frame_id += 1

    h, w, _ = frame.shape

    # Convert to RGB for Mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    #ai detect
    results = detector.detect_for_video(mp_image, frame_id)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks[0]
        
        for p in POSE_POINTS:
            lm = landmarks[p]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, POINT_COL, -1)
        for p1, p2 in POSE_CONNECTIONS:
            x1, y1 = int(landmarks[p1].x * w), int(landmarks[p1].y * h)
            x2, y2 = int(landmarks[p2].x * w), int(landmarks[p2].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), CONNECTION_COL, 2)
        cv2.imshow("Pose detection", frame)
        end_time = time()
        print(f"Whole image making time: {end_time - start_time:.4f} seconds")

        if cv2.waitKey(1) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          break
        
def image_detect(file_path):   #for testing
  # Load the input image.
  image = cv2.imread(file_path)
  h, w, _ = image.shape

  # Convert to RGB for Mediapipe
  rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

  #ai detect pose
  results = detector.detect_for_video(mp_image, 1)

  if results.pose_landmarks:
        landmarks = results.pose_landmarks[0]
        for p in POSE_POINTS:
            lm = landmarks[p]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 4, (0,255,0), -1)
        for p1, p2 in POSE_CONNECTIONS:
            x1, y1 = int(landmarks[p1].x * w), int(landmarks[p1].y * h)
            x2, y2 = int(landmarks[p2].x * w), int(landmarks[p2].y * h)
            cv2.line(image, (x1, y1), (x2, y2), (255,0,0), 2)

        cv2.imshow("Pose detection", image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
          cv2.destroyAllWindows()



#testing
if __name__ == "__main__":
  camera_feed_detect()
  #image_detect("images/example_image.jpg")