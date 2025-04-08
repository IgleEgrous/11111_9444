import os
import shutil

'''
This script restructures the dataset by organizing images into directories based on their magnification level and tumor differentiation stage.'''

# Define the source and destination base paths
source_base = './11111_9444/EBH-HE-IDS/ColHis-IDS'
destination_base = './ColHis-IDS_restructured'

# Walk through the source base directory
for tumor_differentiation_stage in os.listdir(source_base):
    tumor_differentiation_stage_path = os.path.join(source_base, tumor_differentiation_stage)

    # Check if it's a directory
    if os.path.isdir(tumor_differentiation_stage_path):
        for date in os.listdir(tumor_differentiation_stage_path):
            date_path = os.path.join(tumor_differentiation_stage_path, date)

            # Check if it's a directory
            if os.path.isdir(date_path):
                for magnification in os.listdir(date_path):
                    magnification_path = os.path.join(date_path, magnification)

                    # Check if it's a directory
                    if os.path.isdir(magnification_path):
                        for img_file in os.listdir(magnification_path):
                            img_file_path = os.path.join(magnification_path, img_file)

                            # Check if it is a file
                            if os.path.isfile(img_file_path):
                                # Create the corresponding destination directory based on magnification level and tumor differentiation stage
                                destination_tumor_differentiation_stage_dir = os.path.join(destination_base, magnification, tumor_differentiation_stage)
                                os.makedirs(destination_tumor_differentiation_stage_dir, exist_ok=True)

                                # Ensure unique filenames by appending the relative path to the filename
                                unique_filename = f"{date}_{img_file}"
                                destination_file_path = os.path.join(destination_tumor_differentiation_stage_dir, unique_filename)

                                # Copy the image to the new location
                                shutil.copy(img_file_path, destination_file_path)
                                print(f"Copied {img_file} to {destination_file_path}")