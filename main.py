import os
import argparse
from src.ProcessImages import ProcessImages
from src.VideoToImages import VideoToImages
from src.ImageMatching import StitchDirectory  # Make sure to update your import path accordingly


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
                        help='Do grayscale conversion (0 or 1)', default=1)
    parser.add_argument('--sim', type=float,
                        help='Similarity level for removing duplicates (0.0 to 1.0)', default=0.8)
    parser.add_argument('--unique', type=int,
                        help='Removes duplicate images (0 or 1)', default=1)
    parser.add_argument('--start-time', type=str,
                        help='Start capturing at a specified time. Format: MINUTES:SECONDS', default=None)
    parser.add_argument('--end-time', type=str,
                        help='End capturing screenshots at this time. Format: MINUTES:SECONDS ', default=None)
    parser.add_argument('--stitch', type=int,
                        help='Enable stitching of images (0 or 1)', default=1)
    parser.add_argument('--overlap', type=int,
                        help='Overlap region for stitching', default=300)
    parser.add_argument('--threshold', type=int,
                        help='Threshold for formatting stitched image', default=1000)
    parser.add_argument('--padding', type=int,
                        help='Padding for formatted stitched image', default=10)
    parser.add_argument('--pdf', type=int,
                        help='Convert stitched image to PDF (0 or 1)', default=0)

    args = parser.parse_args()

    youtube_url = args.youtube_url
    interval = args.interval
    output_base_dir = "./outputs"
    output_name = args.output_name

    video_to_images = VideoToImages(
        youtube_url, interval, os.path.join(output_base_dir, output_name), output_name, args.start_time, args.end_time)

    raw_screenshots_dir = video_to_images.run()

    input_directory = f"{raw_screenshots_dir}"
    image_processor = ProcessImages(input_directory, force_grayscale=bool(args.gray),
                                    similarity_level=float(args.sim), force_unique=bool(args.unique))

    print("Finished processing images...")

    if args.stitch and args.unique:
        stitcher = StitchDirectory(f"{input_directory}_unique", overlap_region=args.overlap)
        stitcher.stitch_images(do_plot=False)

        if args.threshold or args.padding:
            formatted_img = stitcher.format_stitched_image(threshold=args.threshold, padding=args.padding)

            if args.pdf:
                stitcher.convert_to_pdf(formatted_img)

        print("Finished stitching images...")

if __name__ == "__main__":
    main()
