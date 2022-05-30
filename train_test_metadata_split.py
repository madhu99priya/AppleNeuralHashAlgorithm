import os
import random


if __name__ == "__main__":

    train_test_split = 0.7

    img_dir = f"{os.getcwd()}/images"
    filename_list = os.listdir(img_dir)
    subject_list = []
    for filename in filename_list:
        subject = filename.split('_')[0]
        if subject not in subject_list:
            subject_list.append(subject)

    if not os.path.isdir(f"{img_dir}/training"):
        os.mkdir(f"{img_dir}/training")
    if not os.path.isdir(f"{img_dir}/testing"):
        os.mkdir(f"{img_dir}/testing")

    for s in subject_list:
        if random.random() < train_test_split:
            new_dir = "training"
        else:
            new_dir = "testing"

        for s_img in [x for x in filename_list if x.split('_')[0] == s]:
            os.replace(f"{img_dir}/{s_img}", f"{img_dir}/{new_dir}/{s_img}")
