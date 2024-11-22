import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_images(original, processed, title_processed):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title(title_processed)
    plt.axis('off')

    plt.show()

# Load and resize the image
def load_image():
    image = cv2.imread('/content/traffic1.jpg')
    if image is None:
        raise ValueError("Image not found. Please check the path.")
    image = cv2.resize(image, (250, 250))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, image_gray

def simple_thresholding(image, method='binary'):
    if method == 'binary':
        _, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    elif method == 'otsu':
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def adaptive_thresholding(image):
    return cv2.adaptiveThreshold(image, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=11,
                                  C=2)

def region_growing_scratch(image, seed):
    height, width = image.shape
    segmented = np.zeros_like(image)
    region = {seed}

    while region:
        x, y = region.pop()
        if segmented[x, y] == 0:
            segmented[x, y] = 255

            # Check neighbors
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= x + i < height and 0 <= y + j < width:
                        if abs(int(image[x, y]) - int(image[x + i, y + j])) < 15:
                            region.add((x + i, y + j))

    return segmented

def region_growing_flood_fill(image, seed):
    height, width = image.shape
    segmented = np.zeros_like(image)

    # Flood fill algorithm
    cv2.floodFill(segmented, None, seed, 255)

    return segmented

def watershed_segmentation(image):
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    ret, binary_thresh = cv2.threshold(image_gray, 120, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    binary_thresh = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel)

    dist_transform = cv2.distanceTransform(binary_thresh, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    markers = markers.astype(np.int32)
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Mark boundaries in red
    return image

# Main execution
print("\nSelect a segmentation method (or type 'exit' to quit):")
print("1. Global Thresholding")
print("2. Adaptive Thresholding")
print("3. Region Growing")
print("4. Watershed Segmentation")

choice = input("Enter your choice (1/2/3/4): ")

if choice.lower() == 'exit':
    print("Exiting the program.")
else:
    try:
        image, image_gray = load_image()  # Reload image once

        if choice == '1':
            print("Select a method:")
            print("1. Binary Threshold")
            print("2. Otsu's Threshold")
            method_choice = input("Enter your choice (1/2): ")
            if method_choice == '1':
                result = simple_thresholding(image_gray, method='binary')
                show_images(image_gray, result, "Binary Thresholding")
            elif method_choice == '2':
                result = simple_thresholding(image_gray, method='otsu')
                show_images(image_gray, result, "Otsu's Thresholding")
            else:
                print("Invalid choice! Please select 1 or 2.")

        elif choice == '2':
            result = adaptive_thresholding(image_gray)
            show_images(image_gray, result, "Adaptive Thresholding")

        elif choice == '3':
            print("Select a method for Region Growing:")
            print("1. Scratch Method")
            print("2. Flood Fill Method")
            method_choice = input("Enter your choice (1/2): ")

            height, width = image_gray.shape
            middle_x, middle_y = height // 2, width // 2
            print(f"Suggested seed coordinates: ({middle_x}, {middle_y})")
            seed_x = int(input("Enter seed point x-coordinate (e.g., 100): "))
            seed_y = int(input("Enter seed point y-coordinate (e.g., 100): "))

            if method_choice == '1':
                result = region_growing_scratch(image_gray, (seed_x, seed_y))
                show_images(image_gray, result, "Region Growing (Scratch)")
            elif method_choice == '2':
                result = region_growing_flood_fill(image_gray, (seed_x, seed_y))
                show_images(image_gray, result, "Region Growing (Flood Fill)")
            else:
                print("Invalid choice! Please select 1 or 2.")

        elif choice == '4':
            result = watershed_segmentation(image)  # Pass the original image
            show_images(image, result, "Watershed Segmentation")

        else:
            print("Invalid choice! Please select 1, 2, 3, or 4.")

    except ValueError as e:
        print(e)
