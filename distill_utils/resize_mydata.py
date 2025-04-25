import cv2
import os
from concurrent.futures import ThreadPoolExecutor

# 设置resize和crop的目标大小
target_size = (160, 120)
crop_size = (112, 112)

# 指定图片路径
img_dir = "./data/UCF101/rawframes"
save_dir = "./data/UCF101/jpegs_112"

# Function to process each image
def process_image(img_path, subdir, save_dir):
    # 读取图片
    img = cv2.imread(img_path)

    # resize图片
    img_resized = cv2.resize(img, target_size)

    # crop图片
    h, w, _ = img_resized.shape
    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2
    img_cropped = img_resized[top:top + crop_size[0], left:left + crop_size[1]]

    # 保存处理后的图片
    save_subdir = os.path.join(save_dir, subdir)
    os.makedirs(save_subdir, exist_ok=True)
    cv2.imwrite(os.path.join(save_subdir, os.path.basename(img_path)), img_cropped)

# Function to process each subdirectory
def process_subdir(subdir):
    subdir_path = os.path.join(img_dir, subdir)
    if os.path.isdir(subdir_path):
        # Create a thread pool to process images in parallel
        with ThreadPoolExecutor() as executor:
            # Prepare a list of image paths to process
            image_paths = [os.path.join(subdir_path, filename) for filename in os.listdir(subdir_path)
                           if filename.endswith(".jpg") or filename.endswith(".png")]
            
            # Use the executor to process images in parallel
            executor.map(lambda img_path: process_image(img_path, subdir, save_dir), image_paths)

        print("Finished processing {}".format(subdir_path))

# Main function to process all subdirectories
def main():
    subdirs = os.listdir(img_dir)
    with ThreadPoolExecutor() as executor:
        # Process all subdirectories in parallel
        executor.map(process_subdir, subdirs)

    # Print completion message
    print("All images processed!")

if __name__ == "__main__":
    main()
