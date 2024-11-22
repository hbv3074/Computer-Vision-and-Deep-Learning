import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image, width=150, height=150):
    """Resize the image to the specified width and height."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_sobel_filter(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(255 * (sobel_magnitude / np.max(sobel_magnitude)))
    return sobel_x, sobel_y, sobel_magnitude

def apply_average_filter(image):
    return cv2.blur(image, (5, 5))

def apply_median_filter(image):
    return cv2.medianBlur(image, 5)

def display_images(images, titles):
    plt.figure(figsize=(10, 7))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 2, i + 1)
        plt.title(title)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking show
    plt.pause(1)  # Add a small pause to ensure the plot is rendered

def display_original_image(image):
    plt.figure(figsize=(5, 5))
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show(block=False)  # Non-blocking show
    plt.pause(1)  # Add a small pause to ensure the plot is rendered

def main():
    # Load the image
    image_path = input("Enter the path to the image file: ")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Image file not found. Please check the file path.")
        return

    # Resize the image to 150x150 pixels
    image = resize_image(image)

    while True:
        print("\nChoose the operation to perform:")
        print("1. Gaussian Blur")
        print("2. Sobel Filter")
        print("3. Average Filter")
        print("4. Median Filter")
        print("5. Display Original Image")
        print("6. Exit")

        choice = input("Enter your choice (1/2/3/4/5/6): ")

        if choice == '1':
            filtered_image = apply_gaussian_blur(image)
            display_images([image, filtered_image], ["Original Image", "Gaussian Blur"])

        elif choice == '2':
            sobel_x, sobel_y, sobel_magnitude = apply_sobel_filter(image)
            display_images([image, sobel_x, sobel_y, sobel_magnitude], ["Original Image", "Sobel X", "Sobel Y", "Sobel Magnitude"])

        elif choice == '3':
            filtered_image = apply_average_filter(image)
            display_images([image, filtered_image], ["Original Image", "Average Filter"])

        elif choice == '4':
            filtered_image = apply_median_filter(image)
            display_images([image, filtered_image], ["Original Image", "Median Filter"])

        elif choice == '5':
            display_original_image(image)

        elif choice == '6':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
    