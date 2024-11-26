import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)  # Create Gaussian noise
    noisy_image = image.astype(np.float32) + noise  # Add noise to image
    return np.clip(noisy_image, 0, 255).astype(np.uint8)  # Ensure values are valid for image



# Function to add salt-and-pepper noise
def add_salt_and_pepper_noise(image, amount=0.02):
    noisy_image = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)

    # Add salt noise
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper noise
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image


# Function to apply mean filter
def apply_mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))


# Function to apply median filter
def apply_median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)


# Function to apply Gaussian filter
def apply_gaussian_filter(image, kernel_size=3, sigma=0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


# Function to display images
def display_images(images, titles, cmap='gray'):
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Main program
def main():
    # Set the image file name
    image_file = 'image.jpg'

    # Check current working directory and print it
    print("Current Working Directory:", os.getcwd())

    # Load the image (convert to grayscale)
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Image '{image_file}' not found in the current directory. Please ensure the image exists.")
        return

    # Add noise
    gaussian_noisy_image = add_gaussian_noise(image)
    sap_noisy_image = add_salt_and_pepper_noise(image)

    # Apply filters for Gaussian noise
    mean_filtered_gaussian = apply_mean_filter(gaussian_noisy_image)
    gaussian_filtered_gaussian = apply_gaussian_filter(gaussian_noisy_image)

    # Apply filters for salt-and-pepper noise
    median_filtered_sap = apply_median_filter(sap_noisy_image)

    # Display results for Gaussian noise
    display_images(
        [image, gaussian_noisy_image, mean_filtered_gaussian, gaussian_filtered_gaussian],
        ["Original Image", "Gaussian Noisy Image", "Mean Filtered", "Gaussian Filtered"]
    )

    # Display results for salt-and-pepper noise
    display_images(
        [image, sap_noisy_image, median_filtered_sap],
        ["Original Image", "Salt-and-Pepper Noisy Image", "Median Filtered"]
    )


if __name__ == "__main__":
    main()
