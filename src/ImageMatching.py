import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter

class ImageMatching:
    """Matches and stitches two images together"""
    def __init__(self, image_paths, overlap_region=300):
        self.image_paths = image_paths
        self.images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]
        self.overlap_region = min(overlap_region, self.images[0].shape[1], self.images[1].shape[1])
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
            self.images[0] = np.clip(self.images[0] * factor, 0, 255).astype(np.uint8)
        elif brightness2 < brightness1:
            factor = brightness1 / brightness2
            self.images[1] = np.clip(self.images[1] * factor, 0, 255).astype(np.uint8)

    def detect_and_compute_features(self):
        sift = cv2.SIFT_create()
        self.kp1, self.des1 = sift.detectAndCompute(self.img1_overlap, None)
        self.kp2, self.des2 = sift.detectAndCompute(self.img2_overlap, None)

    def match_features(self):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.matches = bf.match(self.des1, self.des2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)

    def find_most_common_shift(self):
        pts1 = np.float32([self.kp1[m.queryIdx].pt for m in self.matches]).reshape(-1, 2)
        pts2 = np.float32([self.kp2[m.trainIdx].pt for m in self.matches]).reshape(-1, 2)
        pts1[:, 0] += (self.images[0].shape[1] - self.overlap_region)
        horizontal_shifts = pts1[:, 0] - pts2[:, 0]
        rounded_shifts = np.round(horizontal_shifts).astype(int)
        shift_counter = Counter(rounded_shifts)
        self.most_common_shift = shift_counter.most_common(3)[0][0]

        filtered_matches = [m for i, m in enumerate(self.matches) if rounded_shifts[i] == self.most_common_shift]
        self.filtered_pts1 = np.float32([self.kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 2)
        self.filtered_pts2 = np.float32([self.kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 2)
        self.filtered_pts1[:, 0] += (self.images[0].shape[1] - self.overlap_region)

    def prepare_features_and_matches(self):
        self.detect_and_compute_features()
        self.match_features()
        self.find_most_common_shift()

    def stitch_images(self):
        self.prepare_features_and_matches()
        
        x1, y1 = self.filtered_pts1[0]
        x2, y2 = self.filtered_pts2[0]

        shift_x = int(x1 - (self.images[0].shape[1] - self.overlap_region + x2))

        height1, width1 = self.images[0].shape
        height2, width2 = self.images[1].shape

        canvas_height = max(height1, height2)
        canvas_width = width1 + width2 - shift_x
        stitched_image = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

        stitched_image[:height1, :width1] = self.images[0]
        stitched_image[:height2, width1 - shift_x:width1 - shift_x + width2] = self.images[1]

        return stitched_image

    def plot_images(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Image 1')
        plt.imshow(self.images[0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Image 2')
        plt.imshow(self.images[1], cmap='gray')
        plt.show()

    def plot_feature_matching(self, do_print=True):
        self.prepare_features_and_matches()

        if do_print:
            print(f"{'Match':<8}{'Image 1 (x, y)':<20}{'Image 2 (x, y)':<20}{'Shift (pixels)':<15}")
            print("-" * 60)
            for i, (pt1, pt2) in enumerate(zip(self.filtered_pts1, self.filtered_pts2)):
                x1, y1 = pt1
                x2, y2 = pt2
                print(f"{i + 1:<8}({x1:>6.1f}, {y1:>6.1f}) - ({x2:>6.1f}, {y2:>6.1f}) = {self.most_common_shift:>5} pixels")

        img_matches = cv2.drawMatches(self.img1_overlap, self.kp1, self.img2_overlap, self.kp2, self.matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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

if __name__ == "__main__":
    image_paths = [r"./outputs/real/real_20240618_213201_unique/screenshot_0001.png",
                   r"./outputs/real/real_20240618_213201_unique/screenshot_0004.png"]

    matcher = ImageMatching(image_paths)
    matcher.plot_images()
    matcher.plot_feature_matching()
    matcher.plot_stitched_image()
