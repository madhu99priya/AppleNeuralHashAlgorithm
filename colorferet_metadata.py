import os
import csv
from xml.dom import minidom


IMAGES = "images"
GENDER = "gender"
RACE = "race"

img_dir = f"{os.getcwd()}/images"
xml_path = f"{os.getcwd()}/colorferet_metadata/ground_truths/xml/subjects.xml"
xml_file = minidom.parse(xml_path)

genders_out_path = f"{os.getcwd()}/colorferet_genders.csv"
races_out_path = f"{os.getcwd()}/colorferet_races.csv"
metadata_out_path = f"{os.getcwd()}/colorferet_metadata.csv"

subject_list = xml_file.getElementsByTagName("Subject")
print(subject_list)

subject_images = {}

for img_file in os.listdir(img_dir):
    subject = img_file.split('_')[0]

    if subject in subject_images:
        subject_images[subject][IMAGES].append(img_file)
    else:
        subject_images[subject] = {IMAGES: [img_file],
                                   GENDER: None,
                                   RACE: None}

for s in subject_list:
    subject = s.attributes["id"].value.replace("cfrS", "")
    subject_images[subject][GENDER] = s.childNodes[1].attributes["value"].value
    subject_images[subject][RACE] = s.childNodes[5].attributes["value"].value

    # print(f"\n{subject}")
    # print(subject_images[subject][IMAGES])
    # print(subject_images[subject][GENDER])
    # print(subject_images[subject][RACE])

# with open(genders_out_path, "w") as csv_file:
#     filewriter = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     filewriter.writerow(["image_name", "gender"])
#     for data in subject_images.values():
#         for img_file in data[IMAGES]:
#             filewriter.writerow([img_file, data[GENDER]])
#
# with open(races_out_path, "w") as csv_file:
#     filewriter = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     filewriter.writerow(["image_name", "race"])
#     for data in subject_images.values():
#         for img_file in data[IMAGES]:
#             filewriter.writerow([img_file, data[RACE]])

with open(metadata_out_path, "w") as csv_file:
    filewriter = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(["image_name", "gender", "race"])
    for data in subject_images.values():
        for img_file in data[IMAGES]:
            filewriter.writerow([img_file, data[GENDER], data[RACE]])
