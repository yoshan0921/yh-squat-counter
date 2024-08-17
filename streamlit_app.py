from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import time
import numpy as np
import mediapipe as mp
import streamlit as st
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class SquatCounter:
    # Constructor
    def __init__(self):
        self.success_count = 0
        self.up = False
        self.down = False
        self.shoulder_hip_ankle_angle = 0
        self.shoulder_knee_footindex_angle = 0
        self.hip_knee_footindex_angle = 0

    def process_pose(self, landmarks):
        # Get the coordinates of the observation body point.
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]
        left_footindex = [
            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
        ]

        # Calculate angles.
        self.shoulder_hip_ankle_angle = self.calculate_angle(
            left_shoulder, left_hip, left_ankle
        )
        self.shoulder_knee_footindex_angle = self.calculate_angle(
            left_shoulder, left_knee, left_footindex
        )
        self.hip_knee_footindex_angle = self.calculate_angle(
            left_hip, left_knee, left_footindex
        )

        # Count squats if posture is correct.
        if 170 < self.shoulder_hip_ankle_angle < 180:
            self.up = True
        if self.up and 80 < self.hip_knee_footindex_angle < 100 and 170 < self.shoulder_knee_footindex_angle < 180:
            self.down = True
        if self.down and 170 < self.shoulder_hip_ankle_angle < 180:
            self.success_count += 1
            self.up = False
            self.down = False

        return (
            self.success_count,
            self.shoulder_hip_ankle_angle,
            self.shoulder_knee_footindex_angle,
            self.hip_knee_footindex_angle,
        )

    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle

        return int(angle)


class VideoProcessor(VideoProcessorBase):
    # Constructor
    def __init__(self):
        self.pose = mp_pose.Pose()
        self.squat_counter = SquatCounter()
        self.message_timestamp = None

    # Each time the WebRTC stream receives a frame, the recv method is called.
    # The received frame is passed as the frame parameter.
    def recv(self, frame):
        # The frames are converted to a NumPy array in BGR24 format
        # ,which is the default color space used by OpenCV.
        img = frame.to_ndarray(format="bgr24")

        # Convert BGR format image to RGB format for use in MediaPipe.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect poses in images using the MediaPipe Pose model.
        results = self.pose.process(img_rgb)

        # Checks to see if a pose landmark has been detected.
        if results.pose_landmarks:
            # Retrieve a list containing all detected posed landmarks
            landmarks = results.pose_landmarks.landmark

            # Update the squat count using the detected pose landmarks.
            success_count, shoulder_hip_ankle_angle, shoulder_knee_footindex_angle, hip_knee_footindex_angle = (
                self.squat_counter.process_pose(landmarks)
            )

            # Draw the detected pose landmarks on the image.
            # Draw all landmarks and connections with default settings
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Display angles near the corresponding joints
            left_knee_x = int(
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * img.shape[1])
            left_knee_y = int(
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * img.shape[0])
            cv2.putText(img, f'{int(hip_knee_footindex_angle)}', (left_knee_x, left_knee_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (103, 51, 246), 2, cv2.LINE_AA)

            # Display the position/posture status (values of self.up and self.down)
            if self.squat_counter.up and not self.squat_counter.down:
                cv2.putText(img, 'Standing Position', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            elif self.squat_counter.down:
                cv2.putText(img, 'Squatting Position', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            # Display the "Nice! Squat Counted" message for 0.5 seconds
            if not self.squat_counter.up and not self.squat_counter.down:
                self.message_timestamp = time.time()
                cv2.putText(img, 'Nice! Squat Counted', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (103, 51, 246), 2, cv2.LINE_AA)

            if self.message_timestamp and time.time() - self.message_timestamp < 0.5:
                cv2.putText(img, 'Nice! Squat Counted', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (103, 51, 246), 2, cv2.LINE_AA)
            else:
                self.message_timestamp = None

        return frame.from_ndarray(img, format="bgr24")


def main():
    st.title("Squat Counter with MediaPipe")

    # Display counters
    col1, col2, col3, col4 = st.columns(4)
    success_placeholder = col1.empty()
    shoulder_hip_ankle_angle_placeholder = col2.empty()
    hip_knee_footindex_angle_placeholder = col3.empty()
    shoulder_knee_footindex_angle_placeholder = col4.empty()

    # Start WebRTC stream
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30, "max": 30},
            },
            "audio": False,
        },
    )

    # Continues looping while the WebRTC stream is playing.
    while webrtc_ctx.state.playing:
        if webrtc_ctx.video_processor:
            success_count = webrtc_ctx.video_processor.squat_counter.success_count
            shoulder_hip_ankle_angle = (
                webrtc_ctx.video_processor.squat_counter.shoulder_hip_ankle_angle
            )
            hip_knee_footindex_angle = (
                webrtc_ctx.video_processor.squat_counter.hip_knee_footindex_angle
            )
            shoulder_knee_footindex_angle = (
                webrtc_ctx.video_processor.squat_counter.shoulder_knee_footindex_angle
            )

            success_placeholder.metric(label="Count", value=success_count)
            shoulder_hip_ankle_angle_placeholder.metric(
                label="Shoulder-Hip-Ankle Angle", value=shoulder_hip_ankle_angle
            )
            hip_knee_footindex_angle_placeholder.metric(
                label="Hip-Knee-Foot Index Angle", value=hip_knee_footindex_angle
            )
            shoulder_knee_footindex_angle_placeholder.metric(
                label="Shoulder-Knee-Foot Index Angle", value=shoulder_knee_footindex_angle
            )

        time.sleep(0.1)


if __name__ == "__main__":
    main()
