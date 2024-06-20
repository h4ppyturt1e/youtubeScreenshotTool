import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter

class ImageMatching:
    def __init__(self, image_paths, overlap_region=300):
        self.image_paths = image_paths
        self.overlap_region = overlap_region
        self.images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]
        self.img1_overlap = self.images[0][:, -self.overlap_region:]
        self.img2_overlap = self.images[1][:, :self.overlap_region]
        self.kp1, self.des1 = None, None
        self.kp2, self.des2 = None, None
        self.matches = None
        self.most_common_shift = None

    def detect_and_compute_features(self):
        orb = cv2.ORB_create()
        self.kp1, self.des1 = orb.detectAndCompute(self.img1_overlap, None)
        self.kp2, self.des2 = orb.detectAndCompute(self.img2_overlap, None)

    def match_features(self):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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

    def stitch_images(self):
        self.detect_and_compute_features()
        self.match_features()
        self.find_most_common_shift()

        height1, width1 = self.images[0].shape
        height2, width2 = self.images[1].shape
        overlap_width = int(self.most_common_shift)

        canvas_height = max(height1, height2)
        canvas_width = width1 + width2 - overlap_width
        stitched_image = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

        stitched_image[:height1, :width1] = self.images[0]
        stitched_image[:height2, width1 - overlap_width:] = self.images[1]

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

    def plot_feature_matching(self):
        self.detect_and_compute_features()
        self.match_features()
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
