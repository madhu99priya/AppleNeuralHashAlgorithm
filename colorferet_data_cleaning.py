import os


if __name__ == "__main__":

    img_dir = f"{os.getcwd()}/images"
    for subject_dir_name in os.listdir(img_dir):
        print(f"{subject_dir_name}")
        subject_dir = f"{img_dir}/{subject_dir_name}"
        for img_file in os.listdir(subject_dir):
            file_path = f"{subject_dir}/{img_file}"
            if 'f' not in img_file:
                os.remove(file_path)
