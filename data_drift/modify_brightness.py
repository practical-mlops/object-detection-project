from PIL import Image, ImageEnhance
import os


def adjust_brightness(input_dir, output_dir, unmodified_dir, brightness_factor=1.5):
    # Create the output directories and their corresponding label directories
    for directory in [output_dir, unmodified_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Create labels directory
        labels_dir = directory.replace("images", "labels")
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)

    # Get a list of all image files in the input directory
    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.endswith(("png", "jpg", "jpeg", "bmp", "tiff", "tif"))
    ]

    # Process the first 100 images and their labels
    for i, image_file in enumerate(image_files[:100]):
        try:
            # Handle image processing
            img_path = os.path.join(input_dir, image_file)
            img = Image.open(img_path)

            # Copy original image and label to unmodified directory
            unmodified_path = os.path.join(
                unmodified_dir, f"{os.path.splitext(image_file)[0]}.tif"
            )
            img.save(unmodified_path, format="TIFF")

            # Copy corresponding label file
            label_filename = f"{os.path.splitext(image_file)[0]}.txt"
            input_label_path = os.path.join(
                input_dir.replace("images", "labels"), label_filename
            )
            if os.path.exists(input_label_path):
                unmodified_label_path = os.path.join(
                    unmodified_dir.replace("images", "labels"), label_filename
                )
                with open(input_label_path, "r") as src, open(
                    unmodified_label_path, "w"
                ) as dst:
                    dst.write(src.read())

            # Process and save modified image
            enhancer = ImageEnhance.Brightness(img)
            img_enhanced = enhancer.enhance(brightness_factor)
            output_path = os.path.join(
                output_dir, f"{os.path.splitext(image_file)[0]}.tif"
            )
            img_enhanced.save(output_path, format="TIFF")

            # Copy label file to modified directory
            output_label_path = os.path.join(
                output_dir.replace("images", "labels"), label_filename
            )
            with open(input_label_path, "r") as src, open(
                output_label_path, "w"
            ) as dst:
                dst.write(src.read())

            print(f"Processed {image_file} -> {output_path}")
            print(f"Copied original to {unmodified_path}")
            print(f"Copied labels to both directories")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")


# Define directories
input_directory = "DATA/images"
output_directory = "DATA/modified_brightness/images"
unmodified_directory = "DATA/unmodified_brightness/images"

# Adjust brightness for images in the directory
adjust_brightness(
    input_directory, output_directory, unmodified_directory, brightness_factor=1.5
)
