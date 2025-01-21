import os
import cv2
from pytubefix import YouTube
from datetime import datetime


class VideoToImages:
    def __init__(self, youtube_url, interval, output_base_dir, output_name, start_time, end_time):
        self.youtube_url = youtube_url
        self.interval = interval
        self.output_base_dir = output_base_dir
        self.output_name = output_name
        self.video_path = None
        self.start_time = self.human_readable_length(start_time) 
        self.end_time = self.human_readable_length(end_time)
        

    def create_output_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(
            self.output_base_dir, f"{self.output_name}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    def download_video(self):
        yt = YouTube(self.youtube_url, use_po_token=True)
        yt.po_token_verifier
        video = yt.streams.filter(
            progressive=True, file_extension='mp4').first()
        self.video_path = video.download(filename='temp_video.mp4')
        return self.video_path

    def capture_screenshots(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_interval = int(fps * self.interval)

		# Set the start time of the video
        if self.start_time:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_time * 1000)

        frame_count = 0
        screenshot_count = 0

		# Set the end time of the video if specified, otherwise capture all frames
        while cap.isOpened() and (self.end_time is None or cap.get(cv2.CAP_PROP_POS_MSEC) < self.end_time * 1000):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                screenshot_path = os.path.join(
                    self.output_dir, f"screenshot_{screenshot_count:04d}.png")
                cv2.imwrite(screenshot_path, frame)
                screenshot_count += 1

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
    
    
    def human_readable_length(self,time: str):

        if time is None:
            return None

        time = time.split(":")

        if len(time) == 1:
            return int(time[0])

        if len(time) == 2:
            return int(time[0]) * 60 + int(time[1])

        if len(time) == 3:
            return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])

        return None



    def cleanup(self):
        if self.video_path and os.path.exists(self.video_path):
            os.remove(self.video_path)

    def run(self):
        self.download_video()
        self.create_output_directory()
        self.capture_screenshots()
        self.cleanup()
        return self.output_dir



if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=ZrV8YYwKvSs"
    interval = 5
    output_base_dir = "./outputs"
    output_name = "screenshot_directory"

    video_to_images = VideoToImages(
        youtube_url, interval, output_base_dir, output_name)
    video_to_images.run()
