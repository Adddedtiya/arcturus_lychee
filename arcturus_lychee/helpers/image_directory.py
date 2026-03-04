import os

# Valid Image Extensions
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', 
    '.ppm', '.PPM', 
    '.bmp', '.BMP',
    '.tif', '.TIF', 
    '.tiff', '.TIFF',
]

def _is_ext_image_file(filename : str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def scan_directory_for_images(root_dir : str) -> list[str]:
    
    # store the files here !
    image_files = []

    # sanity check
    assert os.path.isdir(root_dir), f'The Path "{root_dir}" is not a valid directory'

    # scan the directories
    for root, _, fnames in os.walk(root_dir):
        for fname in fnames:
            if _is_ext_image_file(fname):
                fpath = os.path.join(root, fname)
                fpath = os.path.abspath(fpath)
                image_files.append(fpath)
    
    # sort the directory and return
    image_files.sort()
    return image_files