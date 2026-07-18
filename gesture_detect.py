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

detector = PoseLandmarker.create_from_options(options)


# Skeleton points to draw
POSE_POINTS = [
    0, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28
]
POINT_COL = (255,255,0)
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
CONNECTION_COL = (255,120,0)


##FUNCTIONS FOR POSE DETECTION AND DISPLAYING DRAWN SKELETON
def detect_pose(image, frame_id):
   # Convert to RGB for Mediapipe
   rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

   # detect
   results = detector.detect_for_video(mp_image, frame_id)

   if results.pose_landmarks:
     landmarks = results.pose_landmarks[0]
   else:
     landmarks = None
   return landmarks

def draw_landmarks(image, landmarks):
  if landmarks == None:
    return image

  h, w, _ = image.shape

  for p in POSE_POINTS:
    lm = landmarks[p]
    x, y = int(lm.x * w), int(lm.y * h)
    cv2.circle(image, (x, y), 4, POINT_COL, -1)
    for p1, p2 in POSE_CONNECTIONS:
      x1, y1 = int(landmarks[p1].x * w), int(landmarks[p1].y * h)
      x2, y2 = int(landmarks[p2].x * w), int(landmarks[p2].y * h)
      cv2.line(image, (x1, y1), (x2, y2), CONNECTION_COL, 2)
  return image




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

if __name__ == "__main__":
  camera_feed_detect()
  #image_detect("images/example_image.jpg")