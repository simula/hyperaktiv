import glob
import os
import argparse

parser = argparse.ArgumentParser(description="Script to combnie all patient features into one file.")

parser.add_argument("-i", "--input-dir", type=str, default="C:\\Users\\Steven\\github\\hyperaktiv\\data\\features")
parser.add_argument("-o", "--output-file", type=str, default="features.csv")

def combine_patient_features(input_dir, output_file):
    
    data = [ ]

    for index, filepath in enumerate(list(glob.glob(os.path.join(input_dir, "*.csv")))):          
        record_id = str(int(os.path.splitext(os.path.basename(filepath))[0].split("_")[1]))
        with open(filepath) as f:
            lines = f.read().splitlines()
            if index == 0:
                data.append(["ID", *lines[0].split(";")])
            data.append([record_id, *lines[1].split(";")])

    with open(output_file, "w") as f:
        for line in data:
            f.write(";".join(line) + "\n")

if __name__== "__main__":

    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_file = args.output_file

    combine_patient_features(input_dir, output_file)