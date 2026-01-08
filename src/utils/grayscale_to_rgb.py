import os
from PIL import Image
import argparse

def convert_grayscale_to_rgb(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_filepath = os.path.join(input_folder, filename)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                grayscale_image = Image.open(input_filepath)

                if grayscale_image.mode != 'L':
                    print(f"Skipping {filename}: Not a grayscale image")
                    continue

                rgb_image = grayscale_image.convert('RGB')

                output_filepath = os.path.join(output_folder, filename)
                rgb_image.save(output_filepath)

                print(f"Converted and saved: {filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images from grayscale to rgb.")
    
    parser.add_argument("--input_dir",
                        type=str,
                        required=True, 
                        help="Path of the input directory.")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True, 
                        help="Path of the output directory.")

    args = parser.parse_args()

    input_folder = args.input_dir
    output_folder = args.output_dir
    
    convert_grayscale_to_rgb(input_folder, output_folder)
