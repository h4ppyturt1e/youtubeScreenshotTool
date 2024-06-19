import os
import argparse
from ProcessImages import ProcessImages
from VideoToImages import VideoToImages


def main():
    parser = argparse.ArgumentParser(
        description='Process video to images and then sanitize images.')
    parser.add_argument('youtube_url', type=str,
                        help='URL of the YouTube video')
    parser.add_argument('interval', type=int,
                        help='Interval in seconds for capturing screenshots')
    parser.add_argument('output_name', type=str,
                        help='Name of the output directory')
    parser.add_argument('--gray', type=int,
                        help='Do grayscale conversion (0 or 1)')
    parser.add_argument('--sim', type=float,
                        help='Similarity level for removing duplicates (0.0 to 1.0)', default=0.8)
    parser.add_argument('--unique', type=int,
                        help='Removes duplicate images (0 or 1)')

    args = parser.parse_args()

    youtube_url = args.youtube_url
    interval = args.interval
    output_base_dir = "./outputs"
    output_name = args.output_name

    video_to_images = VideoToImages(
        youtube_url, interval, os.path.join(output_base_dir, output_name), output_name)
    raw_screenshots_dir = video_to_images.run()

    input_directory = f"{raw_screenshots_dir}"
    image_processor = ProcessImages(input_directory, force_grayscale=bool(args.gray),
                                    similarity_level=float(args.sim), force_unique=bool(args.unique))

    print("Finished processing images...")

if __name__ == "__main__":
    main()
