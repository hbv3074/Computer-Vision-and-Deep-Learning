import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def feature_detection(image, method='ORB'):
    """Detect keypoints and descriptors using the selected feature detection method."""
    if method.upper() == 'ORB':
        detector = cv2.ORB_create()
    elif method.upper() == 'SIFT':
        detector = cv2.SIFT_create()
    else:
        print(f"Unsupported feature detection method: {method}. Using ORB as default.")
        detector = cv2.ORB_create()

    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2, method='FLANN'):
    """Match descriptors between two images using the selected matcher."""
    if method.upper() == 'FLANN':
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif method.upper() == 'BF':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        print(f"Unsupported matching method: {method}. Using FLANN as default.")
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    if method.upper() == 'FLANN':
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    else:
        matches = matcher.match(descriptors1, descriptors2)
        good_matches = sorted(matches, key=lambda x: x.distance)[:50]

    return good_matches

def compute_affine_transform(keypoints1, keypoints2, matches, ransac=True):
    """Compute the affine transformation matrix using good matches."""
    if len(matches) > 3:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        if ransac:
            affine_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        else:
            affine_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        return affine_matrix
    else:
        print("Error: Not enough matches found for affine transformation.")
        return None

def apply_affine_transform(image, affine_matrix):
    """Apply the affine transformation to an image."""
    height, width = image.shape[:2]
    aligned_img = cv2.warpAffine(image, affine_matrix, (width, height))
    return aligned_img

def display_images_side_by_side(img1, img2, title1='Reference Image', title2='Aligned Image'):
    """Display the original and aligned images side by side using Matplotlib."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_matched_features(img1, img2, keypoints1, keypoints2, matches):
    """Display the matched features between the two images."""
    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 8))
    plt.imshow(matched_img)
    plt.title('Matched Features')
    plt.axis('off')
    plt.show()

def display_affine_matrix(affine_matrix):
    """Display the affine transformation matrix."""
    if affine_matrix is not None:
        print("\nAffine Transformation Matrix:")
        print(affine_matrix)
    else:
        print("Affine transformation matrix is not available.")

def load_image(path):
    """Load an image from the given path."""
    img_path = Path(path)
    if not img_path.exists():
        print(f"Error: Image file not found at {path}")
        return None
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to read image at {path}")
    return img

def main_menu():
    """Menu-driven interface for the image registration program."""
    print("\n----- Image Registration Menu -----")
    print("1. Load Images")
    print("2. Detect Features")
    print("3. Match Features")
    print("4. Estimate Affine Transformation")
    print("5. Apply Affine Transformation")
    print("6. Display Aligned Image")
    print("7. Display Matched Features")
    print("8. Display Affine Transformation Matrix")
    print("0. Exit")

def main():
    img1, img2 = None, None
    keypoints1, descriptors1, keypoints2, descriptors2 = None, None, None, None
    good_matches = None
    affine_matrix = None
    aligned_img = None

    while True:
        main_menu()
        choice = input("Enter your choice: ")

        if choice == '1':
            img1_path = input("Enter path for Image 1 (reference image): ")
            img2_path = input("Enter path for Image 2 (image to align): ")
            img1 = load_image(img1_path)
            img2 = load_image(img2_path)
            if img1 is not None and img2 is not None:
                print("Images loaded successfully.")

        elif choice == '2':
            if img1 is not None and img2 is not None:
                method = input("Choose feature detection method (ORB/SIFT): ")
                keypoints1, descriptors1 = feature_detection(img1, method)
                keypoints2, descriptors2 = feature_detection(img2, method)
                print(f"Feature detection completed. Found {len(keypoints1)} keypoints in Image 1 and {len(keypoints2)} keypoints in Image 2.")
            else:
                print("Please load the images first.")

        elif choice == '3':
            if descriptors1 is not None and descriptors2 is not None:
                match_method = input("Choose matching method (FLANN/BF): ")
                good_matches = match_features(descriptors1, descriptors2, match_method)
                print(f"Found {len(good_matches)} good matches.")
            else:
                print("Please detect features first.")

        elif choice == '4':
            if good_matches is not None and len(good_matches) > 3:
                use_ransac = input("Use RANSAC for robust estimation? (y/n): ").lower() == 'y'
                affine_matrix = compute_affine_transform(keypoints1, keypoints2, good_matches, ransac=use_ransac)
                if affine_matrix is not None:
                    print("Affine transformation matrix estimated successfully.")
            else:
                print("Please match features first or check the number of matches.")

        elif choice == '5':
            if affine_matrix is not None:
                aligned_img = apply_affine_transform(img2, affine_matrix)
                print("Affine transformation applied.")
            else:
                print("Please estimate the affine transformation first.")

        elif choice == '6':
            if aligned_img is not None:
                display_images_side_by_side(img1, aligned_img)
            else:
                print("Please apply the affine transformation first.")

        elif choice == '7':
            if good_matches is not None:
                display_matched_features(img1, img2, keypoints1, keypoints2, good_matches)
            else:
                print("Please match features first.")

        elif choice == '8':
            display_affine_matrix(affine_matrix)

        elif choice == '0':
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()