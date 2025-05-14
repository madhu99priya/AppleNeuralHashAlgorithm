import os
import sys
import math
import operator
import itertools
import collections
import json
import csv
import Levenshtein
import numpy as np
from PIL import Image
from onnxruntime import InferenceSession
from xml.dom import minidom
import matplotlib.pyplot as plt

from mtcnn import MTCNN #My changes

# Initialize MTCNN once 
detector = MTCNN() #My changes


# Enable colored output for Windows
if sys.platform in ["cygwin", "win32"]:
    import colorama

    colorama.init()

# ANSI text formatting
END = "\x1b[0m"
BOLD = "\x1b[1m"
RED = "\x1b[31m"
BRIGHT_RED_BG = "\x1b[101m"

# Sorting alternatives for Experiment 2
DIST_SAME = 0
DIST_OTHERS = 1
SD_SAME = 2
SD_OTHERS = 3

# Constants for getting closest/furthest Hamming Distance
CLOSEST = 0
FURTHEST = 1


class NeuralHash:
    def __init__(self):
        # Load ONNX model
        self.model_path = f"{os.getcwd()}/model/model.onnx"
        self.session = InferenceSession(self.model_path)

        # Load output hash matrix
        seed1_path = f"{os.getcwd()}/model/neuralhash_128x96_seed1.dat"
        self.seed1 = open(seed1_path, 'rb').read()[128:]
        self.seed1 = np.frombuffer(self.seed1, dtype=np.float32)
        self.seed1 = self.seed1.reshape([96, 128])

    def calculate_neuralhash(self, image_path):
        """Calculate neuralhash of the image at image_path"""

        arr = self.im2array(image_path)

        # Run model
        inputs = {self.session.get_inputs()[0].name: arr}
        outs = self.session.run(None, inputs)

        # Convert model output to hex hash
        hash_output = self.seed1.dot(outs[0].flatten())
        hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])
        hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)

        return hash_hex, hash_bits

    # @staticmethod
    # def im2array(image_path):
    #     """Preprocess image"""

    #     image = Image.open(image_path).convert('RGB')
    #     image = image.resize([360, 360])
    #     arr = np.array(image).astype(np.float32) / 255.0
    #     arr = arr * 2.0 - 1.0

    #     return arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])

    
    #My Changes
    @staticmethod
    def im2array(image_path):
        """Preprocess image: Detect face with MTCNN, crop, and prepare array"""

        image = Image.open(image_path).convert('RGB')
        img_np = np.array(image)

        # Detect faces
        detections = detector.detect_faces(img_np)

        if detections:
            # Choose the first detected face
            x, y, width, height = detections[0]['box']
            x, y = max(0, x), max(0, y)
            face_image = img_np[y:y+height, x:x+width]
            image = Image.fromarray(face_image)
        else:
            print(f"No face detected in {image_path}, using full image.")

        # Resize cropped face to 360x360
        image = image.resize([360, 360])
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0

        return arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])

class Hamming:
    def __init__(self, split_char, threshold=0, output_format=0, save_dict=False, load_dict=False):
        self.split_char = split_char  # Delimiter for splitting image file names
        self.threshold = threshold  # Threshold hamming distance for considering two images as the same subject
        self.hash_dict = {}  # Dictionary of image hashes. {img: (hash_hex, hash_bin)}
        self.hamming_distances = {}  # Dictionary of hamming distances between image hashes
        self.rates = {}  # Dictionary of subjects' accept and reject rates
        self.output_format = output_format  # Configure output of NeuralHashes to hex or binary. 0 = hex, 1 = bin
        self.save_dict_json = save_dict  # Toggle saving of hash_dict and hamming_distances dicts to json
        self.load_dict_json = load_dict  # Toggle loading of hash_dict and hamming_distances dicts from json
        self.spacing = None  # For aligning output to a grid
        self.hl = None  # Horizontal line
        self.hl_short = None  # Short horizontal line

        #print(f"Using output format: {('hex', 'bin')[self.output_format]}")

    def insert_neuralhash(self, image, nhash):
        """Append neuralhash to self.hash_dict"""

        self.hash_dict[image] = nhash

    def set_printing_params(self):
        """
        Set parameters for aligning output
        according to file names of images
        """

        self.spacing = max([len(i) for i in self.hash_dict]) + 1
        self.hl = f"\n{'-' * (self.spacing + 28 + (self.output_format * 72))}\n"
        self.hl_short = f"\n{'-' * (self.spacing + 28)}\n"

    def save_dict(self, hash_dict="hashes.json", hamming_dict="hamming.json"):
        """Save self.hash_dict and self.hamming_distances to json files"""

        with open(hash_dict, 'w') as handle:
            json.dump(self.hash_dict, handle)

        print(f"Saved hash_dict as {hash_dict}")

        with open(hamming_dict, 'w') as handle:
            json.dump(self.hamming_distances, handle)

        print(f"Saved hamming_distances as {hamming_dict}")

    def load_dict(self, hash_dict="hashes.json", hamming_dict="hamming.json"):
        """Load self.hash_dict and self.hamming_distances from json files"""

        files = os.listdir(os.getcwd())

        for f in files:
            if "hashes" in f and f.endswith(".json"):
                hash_dict = f
            elif "hamming" in f and f.endswith(".json"):
                hamming_dict = f

        with open(hash_dict, 'r') as handle:
            self.hash_dict = json.load(handle)

        print(f"Loaded hash_dict from {hash_dict}")

        with open(hamming_dict, 'r') as handle:
            self.hamming_distances = json.load(handle)

        print(f"Loaded hamming_distances from {hamming_dict}")

    def calculate_hamming_distances(self, max_comp=None):
        """
        Calculate hamming distances of NeuralHashes for
        each image pair and store in self.hamming_distances,
        or only compare up to max_comp if supplied
        """

        if not max_comp:
            max_comp = len(self.hash_dict)

        for n, i in enumerate(self.hash_dict):
            dict_slice = dict(itertools.islice(self.hash_dict.items(), n + 1, max_comp))
            for j in dict_slice:
                hamming_dist_hex = Levenshtein.hamming(self.hash_dict[i][0],
                                                       self.hash_dict[j][0])
                hamming_dist_bin = Levenshtein.hamming(self.hash_dict[i][1],
                                                       self.hash_dict[j][1])
                subject_i = i.split(self.split_char)[0]
                subject_j = j.split(self.split_char)[0]
                same = subject_i == subject_j

                if i in self.hamming_distances:
                    self.hamming_distances[i][j] = (same, hamming_dist_hex, hamming_dist_bin)
                else:
                    self.hamming_distances[i] = {j: (same, hamming_dist_hex, hamming_dist_bin)}

                if j in self.hamming_distances:
                    self.hamming_distances[j][i] = (same, hamming_dist_hex, hamming_dist_bin)
                else:
                    self.hamming_distances[j] = {i: (same, hamming_dist_hex, hamming_dist_bin)}

    def calculate_avg_error_rate(self):
        """Calculate and return the average error rate"""

        FAR_sum = 0
        FRR_sum = 0
        for k, v in self.rates.items():
            FAR_sum += v[4]
            FRR_sum += v[5]

        try:
            FAR = FAR_sum / len(self.rates)
            FRR = FRR_sum / len(self.rates)
        except ZeroDivisionError:
            FAR = 0
            FRR = 1

        return FAR, FRR

    def calculate_rates(self, threshold=None):
        """Calculate FAR and FRR for each subject"""

        if not threshold:
            threshold = self.threshold

        # {subject: [true_positives, false_positives, true_negatives, false_negatives, FAR, FRR, image, {img_files}]}
        self.rates = {}
        for image in self.hamming_distances:
            subject = image.split(self.split_char)[0]
            accepted, rejected, genuine, imposters = self.get_accepted_rejected(image, threshold)
            true_positives = accepted[0]
            false_positives = accepted[1]
            true_negatives = rejected[0]
            false_negatives = rejected[1]

            try:
                FAR = false_positives / (false_positives + true_negatives)
            except ZeroDivisionError:
                FAR = 0
            try:
                FRR = false_negatives / (false_negatives + true_positives)
            except ZeroDivisionError:
                FRR = 1

            if subject in self.rates:
                if self.rates[subject][4] > FAR:
                    self.rates[subject] = [true_positives, false_positives, true_negatives, false_negatives,
                                           FAR, FRR, image, {True: genuine, False: imposters}]
            else:
                self.rates[subject] = [true_positives, false_positives, true_negatives, false_negatives, FAR, FRR,
                                       image, {True: genuine, False: imposters}]

    def get_accepted_rejected(self, image, threshold):
        """
        Take an image filename as input and return a tally of
        accepted and rejected images, as well as image filenames
        """

        accepted = [0, 0]  # [true_positives, false_positives]
        rejected = [0, 0]  # [true_negatives, false_negatives]
        genuine = []  # True positive image files
        imposters = []  # False positive image files
        for img, dist in self.hamming_distances[image].items():
            if dist[1 + self.output_format] <= threshold:
                accepted[not dist[0]] += 1
                if dist[0]:
                    genuine.append(img)
                else:
                    imposters.append(img)
            else:
                # if not dist[0]:  # We set false negatives to 0
                rejected[dist[0]] += 1

        return accepted, rejected, genuine, imposters

    def get_hash_length(self):
        """
        Return the length of the NeuralHashes
        according to self.output_format (hex or bin)
        """

        return len(list(self.hash_dict.values())[0][self.output_format])

    def print_hamming_distance(self, img1, img2):
        """Print hamming distance and neural hashes of input image pair"""

        hash1 = self.hash_dict[img1][self.output_format]
        hash2 = self.hash_dict[img2][self.output_format]
        c1, c2 = self.colored_strings(hash1, hash2)
        distance = Levenshtein.hamming(hash1, hash2)
        self.output_hamming_distances(img1, img2, c1, c2, distance)

    def get_sorted_hamming_dict(self):
        """
        Return sorted dict of unique image pairs by
        hamming distance from lowest to highest
        """

        sorted_dict = {}
        c = 0
        for file_i, dict_i in self.hamming_distances.items():
            for file_j, dist in dict_i.items():
                if not any(collections.Counter((file_i, file_j)) == collections.Counter(i)
                           for i in list(sorted_dict.keys())):
                    sorted_dict[(file_i, file_j)] = dist[1 + self.output_format]
            c += 1
            print(f"{c}/{len(self.hamming_distances)}\t{file_i}")

        return dict(sorted(sorted_dict.items(), key=lambda x: x[1]))

    def print_avg_hamming_distance_and_sd(self, sorting=DIST_SAME):
        """
        For each subject, print average hamming distance
        to other images of the same subject, and average
        hamming distance to images of other subjects.
        Also print standard deviation of hamming distance
        between images of the same subject and different
        subjects.
        :param sorting: int
            Determines how to sort the output
            DIST_SAME sorts on hamming distance between images
                of same subject, from lowest to highest
            DIST_OTHERS sorts on hamming distance between
                different subjects, from lowest to highest
            SD_SAME sorts on standard deviation for images
                of the same subject, from lowest to highest
            SD_OTHERS sorts on standard deviation for images
                of different subjects, from lowest to highest
        """

        if sorting not in [DIST_SAME, DIST_OTHERS, SD_SAME, SD_OTHERS]:
            print(f"{RED}Unexpected value for parameter \"sorting\"\n"
                  f"Expected int in range [0, 3]\n"
                  f"DIST_SAME = 0\n"
                  f"DIST_OTHERS = 1\n"
                  f"SD_SAME = 2\n"
                  f"SD_OTHERS = 3\n"
                  f"Ex: sorting=SD_SAME{END}")
            return

        avg_dist = {}
        sorted_avg_dist_sd = {}
        for file_i, dict_i in self.hamming_distances.items():
            subject_i = file_i.split(self.split_char)[0]
            if subject_i not in avg_dist:
                avg_dist[subject_i] = [0, 0, 0, 0]  # [sum_dist_same, sum_dist_different, total_same, total_different]

            for file_j, dist in dict_i.items():
                # Add hex or bin distance from file_i to file_j to sum_dist_same or sum_dist_different respectively
                avg_dist[subject_i][not dist[0]] += dist[1 + self.output_format]
                # Increment total_same or total_different respectively
                avg_dist[subject_i][(not dist[0]) + 2] += 1

        # Calculate sum of all subjects' distances and totals
        avg_dist["all"] = [sum(i) for i in zip(*avg_dist.values())]

        for subject, tally in avg_dist.items():
            try:
                # (sum_dist_same / total_same, sum_dist_different / total_different)
                sorted_avg_dist_sd[subject] = [tally[0] / tally[2], tally[1] / tally[3]]
            except (IndexError, ZeroDivisionError) as e:
                print(f"{RED}{e} for subject {subject}{END}")

        # Calculate average distance for all images
        sorted_avg_dist_sd["all"] = [avg_dist["all"][0] / avg_dist["all"][2], avg_dist["all"][1] / avg_dist["all"][3]]

        # Build lists of (hamming distance - average hamming distance) ** 2 for each comparison and store in deviation
        deviation = {"all": [[], []]}
        for file_i, dict_i in self.hamming_distances.items():
            subject = file_i.split(self.split_char)[0]
            if subject not in deviation:
                deviation[subject] = [[], []]  # [[distance same], [distance different]]
            for file_j, dist in dict_i.items():
                try:
                    deviation[subject][not dist[0]].append((dist[1 + self.output_format]
                                                            - sorted_avg_dist_sd[subject][not dist[0]]) ** 2)
                    deviation["all"][not dist[0]].append((dist[1 + self.output_format]
                                                          - sorted_avg_dist_sd["all"][not dist[0]]) ** 2)
                except KeyError:
                    pass

        # Calculate standard deviation of hamming distances for same and different subjects
        for subject, dev in deviation.items():
            try:
                variance_same = sum(dev[0]) / len(dev[0])
                variance_different = sum(dev[1]) / len(dev[1])
                sd_same = math.sqrt(variance_same)
                sd_different = math.sqrt(variance_different)
                sorted_avg_dist_sd[subject] += [sd_same, sd_different]
            except ZeroDivisionError:
                pass

        # Sort by value determined by sorting
        sorted_avg_dist_sd = dict(sorted(sorted_avg_dist_sd.items(), key=lambda x: x[1][sorting]))

        # Print output to a grid
        print(self.hl)
        column_titles = "{0}    {1}    {2}    {3}".format("Avg dist self", "\u03C3 self", "Avg dist others",
                                                          "\u03C3 others")
        print("{0:>{x}}    {1}".format("", column_titles, x=len(list(avg_dist.items())[0][0]) + 1))
        print("{0:>6}    {1:>13}    {2:>6}    {3:>15}    {4:>8}".format(
            "All:",
            "{dist:.2f}".format(dist=sorted_avg_dist_sd["all"][DIST_SAME]),
            "{sd:.2f}".format(sd=sorted_avg_dist_sd["all"][SD_SAME]),
            "{dist:.2f}".format(dist=sorted_avg_dist_sd["all"][DIST_OTHERS]),
            "{sd:.2f}".format(sd=sorted_avg_dist_sd["all"][SD_OTHERS])
        ))
        for subject, data in sorted_avg_dist_sd.items():
            if subject == "all":
                continue
            dist_self = "{dist:.2f}".format(dist=(data[DIST_SAME]))
            dist_others = "{dist:.2f}".format(dist=(data[DIST_OTHERS]))
            sd_same = "{sd:.2f}".format(sd=data[SD_SAME])
            sd_others = "{sd:.2f}".format(sd=data[SD_OTHERS])
            print("{0:>6}    {1:>13}    {2:>6}    {3:>15}    {4:>8}".format(
                f"{subject}:", dist_self, sd_same, dist_others, sd_others
            ))

    def print_hamming_distances_sorted(self):
        """Print hamming distance comparison sorted by hamming distance"""

        for img_pair, distance in self.get_sorted_hamming_dict().items():
            hash1 = self.hash_dict[img_pair[0]][self.output_format]
            hash2 = self.hash_dict[img_pair[1]][self.output_format]
            c1, c2 = self.colored_strings(hash1, hash2)
            self.output_hamming_distances(img_pair[0], img_pair[1], c1, c2, distance)

    def print_all_hamming_distances(self):
        """Print all hamming distances of image pairs in self.hamming_distances"""

        for file_i, dict_i in self.hamming_distances.items():
            hash1 = self.hash_dict[file_i][self.output_format]
            for file_j, dist in dict_i.items():
                hash2 = self.hash_dict[file_j][self.output_format]
                hamming_dist = dist[1 + self.output_format]
                c1, c2 = self.colored_strings(hash1, hash2)
                self.output_hamming_distances(file_i, file_j, c1, c2, hamming_dist)

    def compare_hamming_distances(self, image):
        """Compare hamming distances from image to all the other images"""

        hash1 = self.hash_dict[image][self.output_format]
        for file_i, dist in self.hamming_distances[image].items():
            hash2 = self.hash_dict[file_i][self.output_format]
            hamming_dist = dist[1 + self.output_format]
            c1, c2 = self.colored_strings(hash1, hash2)
            self.output_hamming_distances(image, file_i, c1, c2, hamming_dist)

    def output_hamming_distances(self, image_i, image_j, hash_i, hash_j, hamming_distance):
        """Align output to a grid"""

        output0 = "{:>{x}}    {:>{y}}".format(image_i, hash_i, x=self.spacing, y=24)
        output1 = "{:>{x}}    {:>{y}}".format(image_j, hash_j, x=self.spacing, y=24)
        output2 = "{:>{x}}    {:>0}".format("dist", hamming_distance, x=self.spacing)
        print("")
        print(output0)
        print(output1)
        print(output2)

    def print_error_rates_sorted(self, show_image_files=False):
        """
        Print self.rates sorted, containing
        TP, FP, TN, FN, FAR, FRR and image files
        """

        if show_image_files:
            sorted_rates = {k: v for k, v in sorted(self.rates.items(), key=lambda x: x[1][4])}
        else:
            sorted_rates = {k: v[:-2] for k, v in sorted(self.rates.items(), key=lambda x: x[1][4])}

        print(f"Printing error rates for {len(self.rates)} subjects:")
        for subject, rates in sorted_rates.items():
            print(f"{subject}: {rates}")

    @staticmethod
    def colored_strings(str1, str2):
        """
        Return colored strings using ANSI character
        encoding such that matching characters are
        colored and different characters are not
        """

        match_list = Hamming.char_match_list(str1, str2)
        colored_str1 = ""
        colored_str2 = ""
        insert_color = True
        insert_end = False
        for n, match in enumerate(match_list):
            if not match:
                if insert_color:
                    colored_str1 += BRIGHT_RED_BG
                    colored_str2 += BRIGHT_RED_BG
                    insert_color = False
                    insert_end = True
            else:
                if insert_end:
                    colored_str1 += END
                    colored_str2 += END
                    insert_end = False
                    insert_color = True

            colored_str1 += str1[n]
            colored_str2 += str2[n]

        if insert_end:
            colored_str1 += END
            colored_str2 += END

        return colored_str1, colored_str2

    @staticmethod
    def char_match_list(str1, str2):
        """
        Create a binary list comparing two strings
        where 1 represents an identical character
        and 0 represents a different character
        """

        match_list = []
        for i in range(len(str1)):
            if str1[i] == str2[i]:
                match_list.append(1)
            else:
                match_list.append(0)

        return match_list

    def plot_far_frr(self):
        """
        Plot the average FAR and FRR for threshold
        varying from 0 to length of hash
        """

        avg_FAR = []
        avg_FRR = []
        for i in range(self.get_hash_length() + 1):
            self.calculate_rates(threshold=i)
            # self.print_error_rates_sorted(show_image_files=True)
            FAR_i, FRR_i = self.calculate_avg_error_rate()
            avg_FAR.append(FAR_i)
            avg_FRR.append(FRR_i)
            step = "{percent:.2f}".format(percent=(i / self.get_hash_length()))
            print(f"FAR_{step}: {FAR_i}, FRR_{step}: {FRR_i}")

        self.plot_avg_error(avg_FAR, avg_FRR)

    def plot_avg_error(self, FAR, FRR):
        """Plot graph of average error with varying thresholds"""

        x_axis = [i / self.get_hash_length() for i in range(self.get_hash_length() + 1)]
        plt.plot(x_axis, FAR, color="blue", label="FAR")
        plt.plot(x_axis, FRR, color="red", label="FRR")
        plt.xlabel("Threshold")
        plt.ylabel("Error")
        plt.grid(color="gray", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.show()

    def plot_hamming_distances(self, subjects=None):
        """
        Plot graphs of hamming distances between subjects
        :param subjects: str, list, tuple
            String of one subject or list/tuple of strings
            of subjects
        """

        hash_length = self.get_hash_length()
        y_data = self.get_plot_data(subjects=subjects)
        if not y_data:
            return

        for s in y_data:
            if s == "self":
                blue_label = "Same subject"
                red_label = "Different subjects"
            else:
                blue_label = s
                red_label = "Other subjects"
            x_axis = np.linspace(0, hash_length + 1, hash_length + 1)
            plt.fill_between(x_axis, y_data[s][0], 0, color="blue", alpha=0.5, label=blue_label)
            plt.fill_between(x_axis, y_data[s][1], 0, color="red", alpha=0.5, label=red_label)
            plt.xlabel("Hamming Distance")
            plt.ylabel("Proportion of images")
            plt.grid(color="gray", linestyle="--", linewidth=0.5)
            plt.legend()
            plt.show()

    def get_plot_data(self, subjects=None, aggregate=True):
        """
        Create a dictionary containing plot data
        corresponding to distance to other images
        :param subjects: str, list, tuple
            If subjects is None, set subjects to
            dictionary of hamming distances for
            all subjects
        :param aggregate: bool
            Whether to aggregate data for same
            subject into same plot, or use
            a different plot for each image
        """

        if not subjects:
            subjects = self.hamming_distances
        elif isinstance(subjects, (str, list, tuple)):
            subjects = {k: v for k, v in self.hamming_distances.items() if k.split(self.split_char)[0] in subjects}
        else:
            print(f"{RED}Unexpected value for parameter \"subjects\"\n"
                  f"Expected str, list of str, tuple of str\n"
                  f"Ex: subjects=\"00001\", subjects=[\"00001\", \"00002\"]{END}")
            return

        dist_count = {}
        for n, (file_i, dict_i) in enumerate(subjects.items()):
            for file_j, dist in dict_i.items():
                if file_i not in dist_count:
                    dist_count[file_i] = [np.zeros(self.get_hash_length() + 1), np.zeros(self.get_hash_length() + 1)]

                # Increment element corresponding to distance and list
                dist_count[file_i][not dist[0]][dist[1 + self.output_format]] += 1

        if aggregate:
            return self.aggregate_plot_data(dist_count)

        return dist_count

    def aggregate_plot_data(self, counts):
        """
        Aggregates the hamming distance count
        for all images of the same subject
        compared to him-/herself, and others
        """

        y_dict = {"self": [np.zeros(self.get_hash_length() + 1), np.zeros(self.get_hash_length() + 1)]}

        for filename, count in counts.items():
            y_dict["self"][0] = np.add(y_dict["self"][0], count[0])
            y_dict["self"][1] = np.add(y_dict["self"][1], count[1])
            subject_i = filename.split(self.split_char)[0]
            if subject_i not in y_dict:
                y_dict[subject_i] = count
            else:
                for n, array in enumerate(list(count)):
                    y_dict[subject_i][n] = np.add(y_dict[subject_i][n], array)

        for k, v in y_dict.items():
            y_dict[k][0] = self.normalize(v[0])
            y_dict[k][1] = self.normalize(v[1])

        return y_dict

    @staticmethod
    def normalize(arr):
        """Normalize the values in arr to the range [0, 1]"""

        return arr / np.max(arr)

    def count_hamming_distances(self):
        """Count number of comparisons sorted by hamming distance"""

        count_list = [0 for _ in range(self.get_hash_length() + 1)]
        for n, (file_i, dict_i) in enumerate(self.hamming_distances.items()):
            dict_slice = dict(itertools.islice(dict_i.items(), n, len(self.hamming_distances)))
            for file_j, dist in dict_slice.items():
                count_list[dist[1 + self.output_format]] += 1
                if dist[1 + self.output_format] == 0:
                    print(f"Perfect match: {file_i}, {file_j}")

        return count_list

    def get_mismatch(self, img, proximity):
        """
        Return the image of a different
        subject with lowest/highest Hamming
        Distance to img, according to proximity
        """

        op = operator.lt if proximity else operator.gt
        mismatch = ""
        tmp_dist = 0 if proximity else self.get_hash_length()
        try:
            for file_i, dist in self.hamming_distances[img].items():
                if op(tmp_dist, dist[1 + self.output_format]) and not dist[0]:
                    tmp_dist = dist[1 + self.output_format]
                    mismatch = file_i
        except KeyError:
            print(f"{RED}Invalid input image{END}")
            return

        self.print_hamming_distance(img, mismatch)

    def check_order(self):
        """
        Check if insertion order in hamming_distances
        dict is equal for all image keys in the dict
        """

        offset = 0
        s = 0
        for n, file_i in enumerate(self.hamming_distances.keys()):
            if file_i == list(self.hamming_distances.keys())[s]:
                offset += 1
                continue
            file_j = list(list(self.hamming_distances.items())[s][1].items())[n - offset][0]
            if file_i != file_j:
                return False

        return True

    def get_metadata_distribution(self):
        """Get distribution of gender and race in dataset"""

        IMAGES = "images"
        GENDER = "gender"
        RACE = "race"

        gender_count = {"Male": 0,
                        "Female": 0}

        race_count = {"Asian": 0,
                      "Asian-Middle-Eastern": 0,
                      "Asian-Southern": 0,
                      "Black-or-African-American": 0,
                      "Hispanic": 0,
                      "White": 0,
                      "Native-American": 0,
                      "Pacific-Islander": 0,
                      "Other": 0}

        xml_path = f"{os.getcwd()}/colorferet_metadata/ground_truths/xml/subjects.xml"
        xml_file = minidom.parse(xml_path)
        subject_list = xml_file.getElementsByTagName("Subject")
        subject_images = {}

        for img_file in self.hash_dict.keys():
            subject = img_file.split(self.split_char)[0]

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

        for s, data in subject_images.items():
            gender_count[data[GENDER]] += len(data[IMAGES])
            race_count[data[RACE]] += len(data[IMAGES])

        return gender_count, race_count

    def generate_metadataset(self):
        """Generate a csv file containing image NeuralHash, gender and race of the subject, for each image"""

        IMAGES = "images"
        GENDER = "gender"
        RACE = "race"

        xml_path = f"{os.getcwd()}/colorferet_metadata/ground_truths/xml/subjects.xml"
        xml_file = minidom.parse(xml_path)
        metadata_out_path = f"{os.getcwd()}/colorferet_metadataset.csv"
        subject_list = xml_file.getElementsByTagName("Subject")
        subject_images = {}

        for img_file in self.hash_dict.keys():
            subject = img_file.split(self.split_char)[0]

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

        with open(metadata_out_path, "w") as csv_file:
            filewriter = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["image_name", "neuralhash", "gender", "race"])
            for data in subject_images.values():
                for img_file in data[IMAGES]:
                    filewriter.writerow([img_file, self.hash_dict[img_file][1], data[GENDER], data[RACE]])


    def calculate_hamming_distance_between_images(self, hash1, hash2):
   
        # hamming_dist_hex = Levenshtein.hamming(hash1[0], hash2[0])
        hamming_dist_bin = Levenshtein.hamming(hash1[1], hash2[1])

        # print(f"\n🔹 Hamming Distance (Hex): {hamming_dist_hex}")
        print(f"\n🔹 Hamming Distance (Binary): {hamming_dist_bin}\n")

        return  hamming_dist_bin
    
# if __name__ == "__main__":

#     #file_format = ".ppm"  # File format of images
#     file_format = (".ppm", ".jpg", ".png")  # Allow multiple formats

#     split_char = '_'  # Delimiter for splitting image file names

#     NHash = NeuralHash()
#     # hamming = Hamming(split_char, output_format=0, load_dict=True)
#     hamming = Hamming(split_char, output_format=0, save_dict=True, load_dict=False)

    

#     if not os.path.exists(f"{os.getcwd()}/images"):
#         os.mkdir(f"{os.getcwd()}/images")

#     dir_images = f"{os.getcwd()}/images"
    
#     if os.path.isfile(sys.argv[1]):  # If input is a file, use it directly
#         image_paths = [sys.argv[1]]
#     elif os.path.isdir(sys.argv[1]):  # If input is a directory, process all images inside
#         dir_images = sys.argv[1]
#         image_paths = [os.path.join(dir_images, img) for img in os.listdir(dir_images)]
#     else:
#         print(f"Invalid input: {sys.argv[1]}")
#         sys.exit(1)



#     if hamming.load_dict_json:
#         try:
#             print("correct")
#             hamming.load_dict()
            
            
#         except FileNotFoundError as e:
#             print(f"{RED}Could not find json files containing hashes and hamming distances{END}")
#             sys.exit(1)
#     else:
#         for filepath in image_paths:
#             file = filepath.split('/')[-1]
#             print(f"Processing file: {file}")  # Debugging: Print each file name
#             if not file.endswith(file_format):
#                 print(f"Skipping file {file}, unsupported format.")  # Debugging: Unsupported format message
#                 continue

#     try:
#         print(f"Calculating neural hash for {filepath}")  # Debugging: Before neural hash calculation
#         img = Image.open(filepath)
#         neural_hash = NHash.calculate_neuralhash(filepath)
#         print(f"Neural Hash for {file}: {neural_hash}")  # Print the hash to terminal
#         hamming.insert_neuralhash(file, neural_hash)
#     except Exception as e:
#         print(f"Error processing {filepath}: {e}")  # Debugging: Catch all errors
#         hamming.calculate_hamming_distances()

#     if hamming.save_dict_json:
#         hamming.save_dict()

#     hamming.set_printing_params()

#     # hamming.plot_hamming_distances()             # Experiment 1
#     # hamming.print_avg_hamming_distance_and_sd()  # Experiment 2
#     # hamming.plot_far_frr()                       # Experiment 3

# Allowed file formats
# Allowed file formats
# file_format = (".ppm", ".jpg", ".png")

# split_char = '_'  # Delimiter for splitting image file names

# # Initialize NeuralHash and Hamming classes
# NHash = NeuralHash()
# hamming = Hamming(split_char, output_format=0, save_dict=False, load_dict=False)

# # Ensure correct usage
# if len(sys.argv) not in [2, 3]:
#     print("Usage:")
#     print("  python main.py <image1_path>         # Prints Neural Hash for one image")
#     print("  python main.py <image1_path> <image2_path>  # Prints Neural Hashes and Hamming Distance")
#     sys.exit(1)

# image1_path = sys.argv[1]

# # Verify that the provided file exists
# if not os.path.isfile(image1_path):
#     print(f"Error: File '{image1_path}' does not exist.")
#     sys.exit(1)

# # Process the first image and compute Neural Hash
# def process_image(image_path):
#     file_name = os.path.basename(image_path)
#     try:
#         print(f"\nProcessing file: {file_name}")
#         neural_hash = NHash.calculate_neuralhash(image_path)
#         print(f"🟢 Neural Hash for {file_name}: {neural_hash}")
#         return neural_hash
#     except Exception as e:
#         print(f"Error processing {file_name}: {e}")
#         sys.exit(1)

# # Compute and print Neural Hash for first image
# hash1 = process_image(image1_path)

# # If only one image is provided, exit after printing Neural Hash
# if len(sys.argv) == 2:
#     print("\n✅ Process completed.")
#     sys.exit(0)

# # If two images are provided, compute and compare Hamming Distance
# image2_path = sys.argv[2]

# # Verify that the second file exists
# if not os.path.isfile(image2_path):
#     print(f"Error: File '{image2_path}' does not exist.")
#     sys.exit(1)

# # Compute Neural Hash for second image
# hash2 = process_image(image2_path)

# # Compute and print Hamming Distance
# hamming_distance_bin = hamming.calculate_hamming_distance_between_images(hash1, hash2)
# print(f"🔹 Hamming Distance between {os.path.basename(image1_path)} and {os.path.basename(image2_path)}:")
# print(f"   - Binary Distance: {hamming_distance_bin}")

# print("\n✅ Process completed.")

def main():
    file_format = (".ppm", ".jpg", ".png")
    split_char = '_'  # Delimiter for splitting image file names

    # Initialize NeuralHash and Hamming classes
    NHash = NeuralHash()
    hamming = Hamming(split_char, output_format=0, save_dict=False, load_dict=False)

    # Ensure correct usage
    if len(sys.argv) not in [2, 3]:
        print("Usage:")
        print("  python main.py <image1_path>         # Prints Neural Hash for one image")
        print("  python main.py <image1_path> <image2_path>  # Prints Neural Hashes and Hamming Distance")
        return

    image1_path = sys.argv[1]
    if not os.path.isfile(image1_path):
        print(f"Error: File '{image1_path}' does not exist.")
        return

    def process_image(image_path):
        file_name = os.path.basename(image_path)
        try:
            print(f"\nProcessing file: {file_name}")
            neural_hash = NHash.calculate_neuralhash(image_path)
            print(f"🟢 Neural Hash for {file_name}: {neural_hash}")
            return neural_hash
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            return None

    hash1 = process_image(image1_path)

    if len(sys.argv) == 2:
        print("\n✅ Process completed.")
        return

    image2_path = sys.argv[2]
    if not os.path.isfile(image2_path):
        print(f"Error: File '{image2_path}' does not exist.")
        return

    hash2 = process_image(image2_path)
    hamming_distance_bin = hamming.calculate_hamming_distance_between_images(hash1, hash2)
    print(f"🔹 Hamming Distance between {os.path.basename(image1_path)} and {os.path.basename(image2_path)}:")
    print(f"   - Binary Distance: {hamming_distance_bin}")

    print("\n✅ Process completed.")

if __name__ == "__main__":
    main()