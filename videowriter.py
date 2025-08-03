import cv2
import os
from tqdm import tqdm


def create_video_from_images(
    image_folder, output_video_path, frame_rate=25, web_compatible=False
):
    """
    Create video from images with optional web compatibility
    """
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

    # get all image files in the folder
    image_files = [
        f
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1] in valid_extensions
    ]
    image_files.sort()  # sort the files in alphabetical order
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")

    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        raise ValueError(f"Failed to read first image: {first_image_path}")
    height, width, _ = first_image.shape

    # create a video writer
    if web_compatible:
        # Use web-compatible codec settings
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # TODO
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for saving the video
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, frame_rate, (width, height)
    )

    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Failed to read image: {image_path}, skipping.")
            continue
        if image.shape[0] != height or image.shape[1] != width:
            print(f"[WARN] Image {image_path} has different size, resizing.")
            image = cv2.resize(image, (width, height))
        try:
            video_writer.write(image)
        except Exception as e:
            print(f"[ERROR] Failed to write frame {image_path}: {e}")

    # Ensure all frames are flushed and file is finalized
    video_writer.release()
    if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
        raise RuntimeError(
            f"Output video was not created or is empty: {output_video_path}"
        )
    print(f"Video saved at {output_video_path}")
