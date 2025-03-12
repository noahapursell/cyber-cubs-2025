import cv2
import numpy as np
import argparse


def add_pixel_noise(image_path, probability=0.2, output_path="output_noised.png"):
    image = cv2.imread(image_path).astype(
        np.float32)  # Load image as float for blending
    h, w, c = image.shape  # Get image dimensions

    # If probability is 0, save the original image and return
    if probability <= 0:
        cv2.imwrite(output_path, image.astype(np.uint8))
        print(f"No noise applied. Saved original image to {output_path}")
        return

    # If probability is 1, replace the entire image with pure noise
    if probability >= 1:
        noisy_image = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
        cv2.imwrite(output_path, noisy_image)
        print(
            f"Full noise applied. Image is completely randomized and saved to {output_path}")
        return

    # Generate full noise image
    noise = np.random.randint(
        0, 256, (h, w, c), dtype=np.uint8).astype(np.float32)

    # Compute weighted average: (1 - probability) * original + probability * noise
    noisy_image = (1 - probability) * image + probability * noise

    # Convert back to uint8 and save
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy_image)
    print(f"Processed image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--probability", type=float, default=0.2,
                        help="Blending factor: 0 = original, 1 = full noise")
    parser.add_argument(
        "--output", type=str, default="output_noised.png", help="Output image filename")

    args = parser.parse_args()
    add_pixel_noise(args.image_path, args.probability, output_path=args.output)
