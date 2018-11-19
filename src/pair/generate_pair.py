import os
import random
import argparse
import sys

class GeneratePairs:
    """
    Generate the pairs.txt file for applying "validate on LFW" on your own datasets.
    """

    def __init__(self, args):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = args.data_dir
        self.pairs_filepath = args.saved_dir + 'pairs.txt'
        self.repeat_times = int(args.repeat_times)
        self.img_ext = '.png'

    def generate(self):
        # The repeate times. You can edit this number by yourself
        folder_number = self.get_folder_numbers()

        # This step will generate the hearder for pair_list.txt, which contains
        # the number of classes and the repeate times of generate the pair
        if not os.path.exists(self.pairs_filepath):
            with open(self.pairs_filepath,"a") as f:
                f.write(str(self.repeat_times) + "\t" + str(folder_number) + "\n")
        for i in range(self.repeat_times):
            self._generate_matches_pairs()
            self._generate_mismatches_pairs()

    def get_folder_numbers(self):
        count = 0
        for folder in os.listdir(self.data_dir):
            if os.path.isdir(self.data_dir + folder):
                count += 1
        return count

    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        for name in os.listdir(self.data_dir):
            if name == ".DS_Store" or name[-3:] == 'txt':
                continue

            a = []
            for file in os.listdir(self.data_dir + name):
                if file == ".DS_Store":
                    continue
                a.append(file)

            with open(self.pairs_filepath, "a") as f:
                temp = random.choice(a).split("_") # This line may vary depending on how your images are named.
                w = temp[0]
                l = random.choice(a).split("_")[1].lstrip("0").rstrip(self.img_ext)
                r = random.choice(a).split("_")[1].lstrip("0").rstrip(self.img_ext)
                f.write(w + "\t" + l + "\t" + r + "\n")


    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """
        for i, name in enumerate(os.listdir(self.data_dir)):
            if name == ".DS_Store" or name[-3:] == 'txt':
                continue

            remaining = os.listdir(self.data_dir)

            del remaining[i]
            remaining_remove_txt = remaining[:]
            for item in remaining:
                if item[-3:] == 'txt':
                    remaining_remove_txt.remove(item)

            remaining = remaining_remove_txt

            other_dir = random.choice(remaining)
            with open(self.pairs_filepath, "a") as f:
                    file1 = random.choice(os.listdir(self.data_dir + name))
                    file2 = random.choice(os.listdir(self.data_dir + other_dir))
                    f.write(name + "\t" + file1.split("_")[1].lstrip("0").rstrip(self.img_ext) \
                     + "\t" + other_dir + "\t" + file2.split("_")[1].lstrip("0").rstrip(self.img_ext) + "\n")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory with aligned images.')
    parser.add_argument('saved_dir', type=str, help='Directory to save pairs.')
    parser.add_argument('--repeat_times', type=str, help='Repeat times to generate pairs', default=30)
    return parser.parse_args(argv)


if __name__ == '__main__':
    generatePairs = GeneratePairs(parse_arguments(sys.argv[1:]))
    generatePairs.generate()