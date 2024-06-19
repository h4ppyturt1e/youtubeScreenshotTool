from ProcessImages import ProcessImages
from VideoToImages import VideoToImages
import os

def main():
    youtube_url = "https://www.youtube.com/watch?v=ZrV8YYwKvSs"
    interval = 2
    output_base_dir = "./outputs"
    output_name = "remember_me_coco"

    video_to_images = VideoToImages(
        youtube_url, interval, os.path.join(output_base_dir, output_name), output_name)
    raw_screenshots_dir = video_to_images.run()

    input_directory = f"{raw_screenshots_dir}"
    image_processor = ProcessImages(input_directory)

if __name__ == "__main__":
    main()