import os

'''
This script counts the number of .jpg files in a given directory and its subdirectories.
'''

def count_jpgs_in_directory(directory):
    jpg_count = 0

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a .jpg file
            if file.lower().endswith('.jpg'):
                jpg_count += 1
    
    return jpg_count

# print(count_jpgs_in_directory('./ColHis-IDS'))
print(count_jpgs_in_directory('D:/Productivity/github/ColHis-IDS_split/test/200'))