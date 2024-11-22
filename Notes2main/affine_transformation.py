import cv2
import numpy as np
import os

def translate_image(image, tx, ty):
    """Translate the image by (tx, ty)."""
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

def scale_image(image, sx, sy):
    """Scale the image by (sx, sy)."""
    scaled_image = cv2.resize(image, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
    return scaled_image

def rotate_image(image, angle):
    """Rotate the image by angle degrees counterclockwise."""
    rows, cols = image.shape[:2]
    center = (cols // 2, rows // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def shear_image(image, kx, ky):
    """Shear the image by kx in x-direction and ky in y-direction."""
    rows, cols = image.shape[:2]
    shear_matrix = np.float32([[1, kx, 0], [ky, 1, 0]])
    sheared_image = cv2.warpAffine(image, shear_matrix, (cols, rows))
    return sheared_image

def reflect_image(image, axis):
    """Reflect the image across a given axis."""
    if axis == 'x':
        reflected_image = cv2.flip(image, 0)  # Vertical flip
    elif axis == 'y':
        reflected_image = cv2.flip(image, 1)  # Horizontal flip
    elif axis == 'y=x':
        reflected_image = cv2.transpose(cv2.flip(image, 1))  # Flip diagonally
    else:
        print("Invalid axis for reflection.")
        return image
    return reflected_image

def save_image(image, operation):
    """Save the transformed image."""
    filename = f"transformed_{operation}.jpg"
    cv2.imwrite(filename, image)
    print(f"Transformed image saved as {filename}")

def main():
    # Load the input image
    while True:
        image_path = input("Enter the path to the input image: ")
        image = cv2.imread(image_path)

        if image is not None:
            break
        print("Error: Unable to load the image. Please check the path and try again.")

    while True:
        print("\nMenu:")
        print("1. Translate")
        print("2. Scale")
        print("3. Rotate")
        print("4. Shear")
        print("5. Reflect")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            tx = float(input("Enter translation in x (tx): "))
            ty = float(input("Enter translation in y (ty): "))
            result_image = translate_image(image, tx, ty)
            operation = "translate"
        elif choice == '2':
            sx = float(input("Enter scaling factor in x (sx): "))
            sy = float(input("Enter scaling factor in y (sy): "))
            result_image = scale_image(image, sx, sy)
            operation = "scale"
        elif choice == '3':
            angle = float(input("Enter rotation angle in degrees: "))
            result_image = rotate_image(image, angle)
            operation = "rotate"
        elif choice == '4':
            kx = float(input("Enter shear factor in x (kx): "))
            ky = float(input("Enter shear factor in y (ky): "))
            result_image = shear_image(image, kx, ky)
            operation = "shear"
        elif choice == '5':
            axis = input("Enter axis of reflection (x, y, y=x): ")
            result_image = reflect_image(image, axis)
            operation = "reflect"
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
            continue

        # Display the transformed image
        cv2.imshow("Transformed Image", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the transformed image
        save_image(result_image, operation)

if __name__ == "__main__":
    main()