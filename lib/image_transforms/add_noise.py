import cv2
import numpy as np
import argparse


def add_pixel_noise(image_path, probability=0.2, output_path="output_noised.png"):
    # Load image as float32 for blending operations
    image = cv2.imread(image_path).astype(np.float32)
    h, w, c = image.shape

    # If probability is 0, just save the original image (no noise, no blur)
    if probability <= 0:
        cv2.imwrite(output_path, image.astype(np.uint8))
        print(f"No noise applied. Saved original image to {output_path}")
        return

    # If probability is 1, use full noise; otherwise blend original with noise.
    if probability >= 1:
        noisy_image = np.random.randint(
            0, 256, (h, w, c), dtype=np.uint8).astype(np.float32)
    else:
        noise = np.random.randint(
            0, 256, (h, w, c), dtype=np.uint8).astype(np.float32)
        noisy_image = (1 - probability) * image + probability * noise

    # Clip values to ensure they're in valid range and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    # Determine the blur amount based on probability.
    # We use a maximum kernel size of 31 when probability == 1.
    max_kernel_size = 31  # must be odd
    ksize = int(probability * max_kernel_size)
    # Ensure the kernel size is odd and at least 1.
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 1:
        ksize = 1

    # Apply Gaussian blur if kernel size is greater than 1.
    if ksize > 1:
        blurred_image = cv2.GaussianBlur(noisy_image, (ksize, ksize), 0)
    else:
        blurred_image = noisy_image

    cv2.imwrite(output_path, blurred_image)
    print(f"Processed image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--probability", type=float, default=0.2,
                        help="Blending factor: 0 = original, 1 = full noise")
    parser.add_argument("--output", type=str, default="output_noised.png",
                        help="Output image filename")

    args = parser.parse_args()
    add_pixel_noise(args.image_path, args.probability, output_path=args.output)
