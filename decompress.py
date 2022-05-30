import os
import sys
from bz2 import BZ2File


def decompress_bz2(file_path):
    subject, filename = file_path.split('/')[-2], file_path.split('/')[-1]
    new_filename = ".".join(filename.split('.')[:-1])
    subject_dir = f"{os.getcwd()}/images/{subject}"
    new_path = f"{subject_dir}/{new_filename}"
    if not os.path.isdir(f"{os.getcwd()}/images/{subject}"):
        os.mkdir(subject_dir)

    with open(new_path, 'wb') as new_file, BZ2File(file_path, 'rb') as file:
        for data in iter(lambda: file.read(100 * 1024), b''):
            new_file.write(data)


if __name__ == "__main__":

    colorferet_dir = f"{os.getcwd()}/colorferet"
    for sub_dir in os.listdir(colorferet_dir):
        data_dir = f"{colorferet_dir}/{sub_dir}/data/images"
        for subject_dir_name in os.listdir(data_dir):
            if os.path.isdir(f"{os.getcwd()}/images/{subject_dir_name}"):
                continue
            print(f"{subject_dir_name}")
            subject_dir = f"{data_dir}/{subject_dir_name}"
            for bz2_file in os.listdir(subject_dir):
                file_path = f"{subject_dir}/{bz2_file}"
                decompress_bz2(file_path)

    file_dict_colorferet = {}
    for sub_dir in os.listdir(colorferet_dir):
        data_dir = f"{colorferet_dir}/{sub_dir}/data/images"
        for subject_dir_name in os.listdir(data_dir):
            subject_dir = f"{data_dir}/{subject_dir_name}"
            subject_dir_file_count = len(os.listdir(subject_dir))
            file_dict_colorferet[subject_dir_name] = subject_dir_file_count

    file_dict_images = {}
    dir_images = f"{os.getcwd()}/images"
    for _subject_dir_name in os.listdir(dir_images):
        _subject_dir = f"{dir_images}/{_subject_dir_name}"
        _subject_dir_file_count = len(os.listdir(_subject_dir))
        file_dict_images[_subject_dir_name] = _subject_dir_file_count

    if len(file_dict_colorferet) != len(file_dict_images):
        print(f"Colorferet subjects: {len(file_dict_colorferet)}, Image subjects: {len(file_dict_images)}")
        sys.exit(0)

    valid = True

    for subject, images in list(file_dict_colorferet.items()):
        if file_dict_images[subject] != images:
            print(f"\x1b[31mSubject: {subject}, c: {images}, i: {file_dict_images[subject]}\x1b[0m")
            valid = False

    if valid:
        print("Decompression successful!")
    else:
        print("Decompression error")