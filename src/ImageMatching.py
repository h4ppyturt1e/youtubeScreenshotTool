import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class ImageMatching:
    """Matches and stitches two images together"""

    def __init__(self, image_paths, overlap_region=300):
        self.image_paths = image_paths
        self.images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                       for image_path in image_paths]
        if overlap_region == -1:
            self.overlap_region = min(
                self.images[0].shape[1], self.images[1].shape[1])
        else:
            self.overlap_region = min(
                overlap_region, self.images[0].shape[1], self.images[1].shape[1])
        self.brighten_images()
        self.img1_overlap = self.images[0][:, -self.overlap_region:]
        self.img2_overlap = self.images[1][:, :self.overlap_region]
        self.kp1, self.des1 = None, None
        self.kp2, self.des2 = None, None
        self.matches = None
        self.most_common_shift = None
        self.filtered_pts1 = None
        self.filtered_pts2 = None

    def brighten_images(self):
        brightness1 = np.mean(self.images[0])
        brightness2 = np.mean(self.images[1])

        if brightness1 < brightness2:
            factor = brightness2 / brightness1
            self.images[0] = np.clip(
                self.images[0] * factor, 0, 255).astype(np.uint8)
        elif brightness2 < brightness1:
            factor = brightness1 / brightness2
            self.images[1] = np.clip(
                self.images[1] * factor, 0, 255).astype(np.uint8)

    def detect_and_compute_features(self):
        orb = cv2.ORB_create()
        self.kp1, self.des1 = orb.detectAndCompute(self.img1_overlap, None)
        self.kp2, self.des2 = orb.detectAndCompute(self.img2_overlap, None)

    def match_features(self):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.matches = bf.match(self.des1, self.des2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)

    def find_most_common_shift(self):
        pts1 = np.float32(
            [self.kp1[m.queryIdx].pt for m in self.matches]).reshape(-1, 2)
        pts2 = np.float32(
            [self.kp2[m.trainIdx].pt for m in self.matches]).reshape(-1, 2)

        pts1[:, 0] += (self.images[0].shape[1] - self.overlap_region)

        horizontal_shifts = pts1[:, 0] - pts2[:, 0]
        rounded_shifts = np.round(horizontal_shifts).astype(int)
        shift_counter = Counter(rounded_shifts)
        self.most_common_shift = shift_counter.most_common(3)[0][0]

        filtered_matches = [m for i, m in enumerate(
            self.matches) if rounded_shifts[i] == self.most_common_shift]
        self.filtered_pts1 = np.float32(
            [self.kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 2)
        self.filtered_pts2 = np.float32(
            [self.kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 2)

        # translate back to original coordinate system
        self.filtered_pts1[:,
                           0] += (self.images[0].shape[1] - self.overlap_region)

    def prepare_features_and_matches(self):
        self.detect_and_compute_features()
        self.match_features()
        self.find_most_common_shift()

    def stitch_images(self, do_plot=True):
        self.prepare_features_and_matches()

        x1, y1 = self.filtered_pts1[0]
        x2, y2 = self.filtered_pts2[0]

        height1, _ = self.images[0].shape
        height2, _ = self.images[1].shape

        cropped_img1 = self.images[0][:, :int(x1)]
        cropped_img2 = self.images[1][:, int(x2):]

        if do_plot:
            print(
                f"Stitching starts at img1: ({x1:.1f}, {y1:.1f}) and img2: ({x2:.1f}, {y2:.1f})")
            self.plot_images([cropped_img1, cropped_img2])

        canvas_height = max(height1, height2)
        canvas_width = cropped_img1.shape[1] + cropped_img2.shape[1]
        stitched_image = np.zeros(
            (canvas_height, canvas_width), dtype=np.uint8)

        stitched_image[:cropped_img1.shape[0],
                       :cropped_img1.shape[1]] = cropped_img1

        stitched_image[:cropped_img2.shape[0],
                       cropped_img1.shape[1]:] = cropped_img2

        non_empty_columns = np.where(stitched_image.max(axis=0) > 0)[0]
        if non_empty_columns.size:
            max_x = non_empty_columns[-1] + 1
            stitched_image = stitched_image[:, :max_x]

        return stitched_image

    def plot_images(self, images=None):
        if images is None:
            images = self.images

        num_images = len(images)
        fig, axs = plt.subplots(1, num_images, figsize=(12, 6))

        for i in range(num_images):
            axs[i].imshow(images[i], cmap='gray')
            axs[i].set_title(f'Image {i+1}')

        plt.show()

    def plot_feature_matching(self, do_print=True):
        self.prepare_features_and_matches()

        if do_print:
            print(
                f"{'Match':<8}{'Image 1 (x, y)':<20}{'Image 2 (x, y)':<20}{'Shift (pixels)':<15}")
            print("-" * 60)
            for i, (pt1, pt2) in enumerate(zip(self.filtered_pts1, self.filtered_pts2)):
                x1, y1 = pt1
                x2, y2 = pt2
                print(
                    f"{i + 1:<8}({x1:>6.1f}, {y1:>6.1f}) - ({x2:>6.1f}, {y2:>6.1f}) = {self.most_common_shift:>5} pixels")

        filtered_matches = [
            m for i, m in enumerate(self.matches)
            if np.round((self.images[0].shape[1] - self.overlap_region + self.kp1[m.queryIdx].pt[0]) - self.kp2[m.trainIdx].pt[0]).astype(int) == self.most_common_shift
        ]

        img_matches = cv2.drawMatches(self.img1_overlap, self.kp1, self.img2_overlap, self.kp2,
                                      filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        self.plot_images([self.img1_overlap, self.img2_overlap])
        plt.figure(figsize=(15, 7))
        plt.title('Feature Matching')
        plt.imshow(img_matches)
        plt.show()

    def plot_stitched_image(self):
        stitched_image = self.stitch_images()
        plt.figure(figsize=(12, 6))
        plt.title('Stitched Image')
        plt.imshow(stitched_image, cmap='gray')
        plt.show()


class StitchDirectory:
    """Stitches all images in a directory together"""

    def __init__(self, directory, overlap_region=300):
        self.directory = directory
        self.overlap_region = overlap_region
        self.image_paths = sorted([os.path.join(directory, f) for f in os.listdir(
            directory) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Found {len(self.image_paths)} images in {directory}")

    def delete_temp_files(self):
        temp_output_path = os.path.join(self.directory, "temp_stitched.png")
        final_output_path = os.path.join(self.directory, "final_stitched.png")
        for file_path in [temp_output_path, final_output_path]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    # print(f"Removed {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    def stitch_images(self, do_plot=True):
        if len(self.image_paths) < 2:
            print("Not enough images to stitch.")
            return
        
        self.delete_temp_files()

        current_image_path = self.image_paths[0]
        for next_image_path in self.image_paths[1:]:
            matcher = ImageMatching(
                [current_image_path, next_image_path], self.overlap_region)
            stitched_image = matcher.stitch_images(do_plot=do_plot)

            # Save the stitched image temporarily
            temp_output_path = os.path.join(self.directory, "temp_stitched.png")
            cv2.imwrite(temp_output_path, stitched_image)

            current_image_path = temp_output_path

        parent_dir = os.path.abspath(os.path.join(self.directory, os.pardir))
        final_output_path = os.path.join(parent_dir, "final_stitched.png")
        if os.path.exists(final_output_path):
            os.remove(final_output_path)
        os.rename(current_image_path, final_output_path)
        print(f"Final stitched image saved as {final_output_path}")
        
        self.delete_temp_files()


if __name__ == "__main__":
    image_paths = [r"./outputs/real/real_20240618_213201_unique/screenshot_0001.png",
                   r"./outputs/real/real_20240618_213201_unique/screenshot_0004.png"]

    matcher = ImageMatching(image_paths)
    matcher.plot_images()
    matcher.plot_feature_matching()
    matcher.plot_stitched_image()
