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
        self.hip_knee_footindex_angle = 0.0
        self.shoulder_knee_ankle_angle = 0.0

    def process_pose(self, landmarks):
        # Get the coordinates of the observation body point.
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
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
        self.shoulder_knee_ankle_angle = self.calculate_angle(
            left_shoulder, left_knee, left_ankle
        )
        self.hip_knee_footindex_angle = self.calculate_angle(
            left_hip, left_knee, left_footindex
        )

        # Count squats if posture is correct.
        if 170 < self.shoulder_knee_ankle_angle < 180:
            self.up = True
        if self.up and 80 < self.hip_knee_footindex_angle < 100:
            self.down = True
        if self.down and 170 < self.shoulder_knee_ankle_angle < 180:
            self.success_count += 1
            self.up = False
            self.down = False

        return (
            self.success_count,
            self.shoulder_knee_ankle_angle,
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
            # Update the squat count using the detected pose landmarks.
            success_count, hip_knee_footindex_angle, shoulder_knee_ankle_angle = (
                self.squat_counter.process_pose(results.pose_landmarks.landmark)
            )

            # Draw the detected pose landmarks on the image.
            # Draw all landmarks and connections with default settings
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Draw text on the image
            # cv2.putText(img, f'Success: {success_count}', (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.putText(img, f'Hip-Knee-Foot Index Angle: {hip_knee_footindex_angle:.2f}', (10, 110),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(img, f'Shoulder-Knee-Ankle Angle: {shoulder_knee_ankle_angle:.2f}', (10, 150),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return frame.from_ndarray(img, format="bgr24")


def main():
    st.title("Squat Counter with MediaPipe")

    # Display counters
    col1, col2, col3 = st.columns(3)
    success_placeholder = col1.empty()
    hip_knee_footindex_angle_placeholder = col2.empty()
    shoulder_knee_ankle_angle_placeholder = col3.empty()

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
            hip_knee_footindex_angle = (
                webrtc_ctx.video_processor.squat_counter.hip_knee_footindex_angle
            )
            shoulder_knee_ankle_angle = (
                webrtc_ctx.video_processor.squat_counter.shoulder_knee_ankle_angle
            )

            success_placeholder.metric(label="Count", value=success_count)
            hip_knee_footindex_angle_placeholder.metric(
                label="Hip-Knee-Foot Index Angle", value=hip_knee_footindex_angle
            )
            shoulder_knee_ankle_angle_placeholder.metric(
                label="Shoulder-Knee-Ankle Angle", value=shoulder_knee_ankle_angle
            )

        time.sleep(0.1)


if __name__ == "__main__":
    main()
