import os


# The file extensions of image files. The comparison ignores uppercase and lowercase.
IMG_EXTENSIONS : set[str] = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff'}


def _is_ext_image_file(filename : str) -> bool:
    """Return True if the file name has an image extension."""
    return os.path.splitext(filename)[1].lower() in IMG_EXTENSIONS


def scan_directory_for_images(root_dir : str) -> list[str]:
    """Return the sorted absolute paths of all image files in root_dir and its subdirectories."""
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f'The path "{root_dir}" is not a directory.')

    image_files = []
    for root, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            if _is_ext_image_file(file_name):
                image_files.append(os.path.abspath(os.path.join(root, file_name)))

    image_files.sort()
    return image_files
