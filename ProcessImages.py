import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from skimage.metrics import structural_similarity as ssim


class ProcessImages:
    def __init__(self, input_directory):
        self.input_directory = input_directory
        self.original_images = self._get_image_paths()
        self.sanitized_images = []
        self.cropped_images = []
        self.grayscaled_images = []
        self.unique_images = []
        
        self.sanitize_directory()
        self.crop_points = self.select_crop_area()
        if self.crop_points:
            self.crop_all_images()
        self.convert_to_grayscale()
        self.remove_duplicates()
        
    def _get_image_paths(self):
        return [os.path.join(self.input_directory, f) for f in os.listdir(self.input_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def sanitize_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return None

        # check for blank screens
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 10:
            return None

        return image

    def sanitize_directory(self):
        """ Sanitizes images and saves them to a new directory. """
        output_directory = f"{self.input_directory}_sanitized"
        # check if sanitized directory already exists, if so, delete it
        if os.path.exists(output_directory):
            for f in os.listdir(output_directory):
                os.remove(os.path.join(output_directory, f))

        os.makedirs(output_directory, exist_ok=True)
        print(f"Sanitizing images and saving to {output_directory}...")

        self.sanitized_images = []
        for image_path in self._get_image_paths():
            image = self.sanitize_image(image_path)
            if image is not None:
                self.sanitized_images.append((image_path, image))

    def select_crop_area(self):
        def click_event(event, x, y, flags, param):
            adjusted_x = max(border_size, min(
                x, image_with_border.shape[1] - border_size - 1))
            adjusted_y = max(border_size, min(
                y, image_with_border.shape[0] - border_size - 1))
            if event == cv2.EVENT_LBUTTONDOWN:
                crop_points.append((adjusted_x, adjusted_y))
            elif event == cv2.EVENT_MOUSEMOVE and len(crop_points) == 1:
                image_copy = image_with_border.copy()
                cv2.rectangle(
                    image_copy, crop_points[0], (x, y), (0, 255, 0), 2)
                cv2.imshow("Select Crop Area", image_copy)
            elif event == cv2.EVENT_LBUTTONUP and len(crop_points) == 2:
                crop_points.append((adjusted_x, adjusted_y))
                cv2.destroyAllWindows()

        def confirm_crop_points():
            x1, y1 = crop_points[0]
            x2, y2 = crop_points[1]
            x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
            confirmed_image = image_with_border.copy()
            cv2.rectangle(confirmed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Confirm Crop Area", confirmed_image)
            cv2.waitKey(1)  # Allow the image to render

            root = tk.Tk()
            root.withdraw()  # Hide the root window
            result = messagebox.askquestion(
                "Confirm Crop Area", "Are you sure about the selected crop area?", icon='warning')
            cv2.destroyAllWindows()
            root.destroy()

            if result == 'yes':
                return (x-border_size, y-border_size, w, h)
            return None

        crop_points = []
        border_size = 20

        if self.sanitized_images:
            _, image = self.sanitized_images[0]
            if image is None:
                print("Failed to load first image.")
                return None

            while True:
                # Add border around the image
                image_with_border = cv2.copyMakeBorder(
                    image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(255, 255, 255))

                cv2.imshow("Select Crop Area", image_with_border)
                cv2.setMouseCallback("Select Crop Area", click_event)
                cv2.waitKey(0)

                if len(crop_points) >= 2:
                    x1, y1 = crop_points[0]
                    x2, y2 = crop_points[1]
                    x, y, w, h = min(x1, x2) - border_size, min(y1, y2) - \
                        border_size, abs(x2 - x1), abs(y2 - y1)
                    crop_area = confirm_crop_points()
                    if crop_area:
                        return crop_area
                    else:
                        crop_points.clear()
                else:
                    print("Crop area not selected properly, exiting...")
                    exit(1)
        else:
            print("No images found in the directory.")
            return None

    def crop_all_images(self):
        output_directory = f"{self.input_directory}_cropped"
        os.makedirs(output_directory, exist_ok=True)

        for image_path, image in self.sanitized_images:
            x, y, w, h = self.crop_points
            cropped_image = image[y:y+h, x:x+w]
            output_path = os.path.join(
                output_directory, os.path.basename(image_path))
            cv2.imwrite(output_path, cropped_image)
            self.cropped_images.append((output_path, cropped_image))

        print(f"Images cropped and saved to {output_directory}...")
        print(
            "Deleting sanitized images directory...")
        for image_path, _ in self.sanitized_images:
            os.remove(image_path)
        os.rmdir(f"{self.input_directory}_sanitized")

    def convert_to_grayscale(self):
        output_directory = f"{self.input_directory}_grayscale"
        os.makedirs(output_directory, exist_ok=True)

        for image_path, image in self.cropped_images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            output_path = os.path.join(
                output_directory, os.path.basename(image_path))
            cv2.imwrite(output_path, gray)
            self.grayscaled_images.append((output_path, gray))

        print(
            f"Images converted to grayscale and saved to {output_directory}...")
        print(
            "Deleting cropped images directory...")
        for image_path, _ in self.cropped_images:
            os.remove(image_path)
        os.rmdir(f"{self.input_directory}_cropped")

    def remove_duplicates(self, similarity_level=0.8):
        output_directory = f"{self.input_directory}_unique"
        os.makedirs(output_directory, exist_ok=True)

        self.unique_images = []
        for i, (image_path, image) in enumerate(self.grayscaled_images):
            is_duplicate = False
            if self.unique_images:
                last_unique_image = self.unique_images[-1][1]
                similarity_index, _ = ssim(image, last_unique_image, full=True)
                if similarity_index >= similarity_level:
                    is_duplicate = True

            if not is_duplicate:
                output_path = os.path.join(output_directory, os.path.basename(image_path))
                cv2.imwrite(output_path, image)
                self.unique_images.append((output_path, image))

        print(f"Unique images saved to {output_directory}...")
        print("Deleting grayscale images directory...")
        for image_path, _ in self.grayscaled_images:
            os.remove(image_path)
        os.rmdir(f"{self.input_directory}_grayscale")

        print(f"Found {len(self.unique_images)} unique images out of {len(self.original_images)} total images.")

if __name__ == "__main__":
    input_directory = "./outputs/remember_me_coco/remember_me_coco_20240618_201918"
    sanitizer = ProcessImages(input_directory)
    sanitizer.convert_to_grayscale()
    sanitizer.remove_duplicates()