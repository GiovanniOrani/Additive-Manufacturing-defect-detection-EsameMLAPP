from PIL import Image
import os
import subprocess
import shutil
import glob
import random
import argparse
from PIL import Image, ImageEnhance
import random
import json
from tqdm import tqdm

n_image_to_generate = 10

# UTILS FUNCTION___________________________________________________________________

def find_image_with_pattern(folder, pattern_list):
    # Find all the image .png with one of the string in pattern list in the name
    images = dict()
    for pattern_string in pattern_list:
        # For each string in pattern list
        images_for_pattern = []
        pattern = os.path.join(folder, f'*_{pattern_string}_*.png')
        # Find all images which contains string
        images_for_pattern = glob.glob(pattern)
        # add found list to images dictionary
        images[pattern_string] = images_for_pattern
    return images

def add_harmonized_to_filename(filename):
    # Add harmonized to filename, in the same way of ConSinGan script
    name, extension = os.path.splitext(filename)
    new_name = f"{name}_harmonized{extension}"
    return new_name

def create_new_json(path_to_create_in):
    # Create a new blank json for image annotations
    category_dict = [
        { "id": 1, "name": "Hole", "supercategory": "Defect" },
        { "id": 2, "name": "Horizontal", "supercategory": "Defect" },
        { "id": 3, "name": "Spattering", "supercategory": "Defect" },
        { "id": 4, "name": "Vertical", "supercategory": "Defect" },
        { "id": 5, "name": "Incandescence", "supercategory": "Defect" }
    ]

    json_skeleton = {
        "images" : [],
        "annotations": [],
        "category" : category_dict
    }

    with open(path_to_create_in, 'w') as json_file:
        json.dump(json_skeleton, json_file, indent=4)

def create_new_log_json(path_to_create_in):
    # Create a new blank json for log
    counters_dict = {
        "Component" : 0,
        "Hole" : 0,
        "Horizontal" : 0,
        "Spattering" : 0,
        "Vertical" : 0,
        "Incandescence" : 0,
        "Blank" : 0,
        "Skipped" : 0,
    }

    json_name = os.path.join(path_to_create_in,"log.json")

    json_skeleton = {
        "counters" : counters_dict,
        "images" : []
    }

    with open(json_name, 'w') as json_file:
        json.dump(json_skeleton, json_file, indent=4)

def filename_no_extension(path):
    return os.path.splitext(os.path.basename(path))[0]

def find_generated_images(folder,pattern_string):
    # Create pattern for searching PNG
    pattern = os.path.join(folder, f"{pattern_string}*.png")

    # Find all PNG images
    images = glob.glob(pattern)

    # Filter all image with mask in the name
    filtered_images = [img for img in images if 'mask' not in os.path.basename(img).lower()]

    return filtered_images

def find_used_models(folder):
    # Return list of folders that exist in a directory
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def remove_model_from_path(path,modelfolder):
    # Remove model folder from a path which contains it
    parts = path.split(os.sep)

    # Remove model Folder from path
    if modelfolder in parts:
        parts.remove(modelfolder)

    new_path = os.sep.join(parts)
    return new_path

# UTILS FUNCTION___________________________________________________________________END

def paste_multiple_images(background_path, images_paths, output_path, area, mask_output_path, annotations_path, log_path):
    """
    Paste multiple images onto a background within a specified area, ensuring they do not overlap.
    Also saves a single combined mask of all the pasted images.

    :param background_path: Path to the background image.
    :param images_paths: List of paths to the images to be pasted.
    :param output_path: Path to save the resulting image.
    :param area: Tuple (x_min, y_min, x_max, y_max) specifying the area within which to paste the images.
    :param mask_output_path: Path to save the combined mask of the pasted images.
    :param annotations_path: Path to save the annotations json file.
    :param log_path: Path to save the log json file.
    """

    try:
        category_map = {"Component":0,"Hole":1,"Horizontal":2,"Spattering":3,"Vertical":4,"Incandescence":5}

        # defects augmentation parameters
        augmented_size_range = (0.7,1.3)
        augmented_brightness_range= (0.9,1.1)
        augmented_rotation_probability = 0.3
        # this is used twice with 2 indipendent check. the probability of at least 1 trasposition is 0.3
        augmented_transpose_probability = 0.1635 

        # Load the background image
        background = Image.open(background_path)
        background = background.convert("RGB")

        # Set the desired size. This is bounded to the harmonization model which is trained with a size of 640x512
        desired_size = (640,512)
        # Compute factor to scale the image to desired size
        desired_size_factor = tuple(x / y for x, y in zip(background.size, desired_size))
        # Resize base image to desired factor
        background = background.resize(desired_size, Image.LANCZOS)

        # Create an empty combined mask for the background
        combined_mask = Image.new("L", background.size, 0)

        # Calculate the dimensions of the area scaled with desired size
        x_min, y_min, x_max, y_max = area
        x_min =(int)(x_min / desired_size_factor[0])
        y_min =(int)(y_min / desired_size_factor[1])
        x_max =(int)(x_max / desired_size_factor[0])
        y_max =(int)(y_max / desired_size_factor[1])
        area_width = (x_max - x_min)
        area_height = (y_max - y_min)

        # Keep track of occupied regions
        occupied_regions = []

        # Read annotations JSON file
        bg_width, bg_height = background.size
        with open(annotations_path, 'r') as file:
            json_data = json.load(file)

        # Add image information to JSON file
        image_id = len(json_data["images"]) + 1

        json_data["images"].append({
            "id": image_id,
            "file_name": os.path.basename(output_path),
            "width": bg_width,
            "height": bg_height
        })

        # Open_json_log
        with open(log_path, 'r') as file:
            json_log = json.load(file)

        json_log['images'].append({
            "id": image_id,
            "file_name": os.path.basename(output_path),
            "background": os.path.basename(background_path),
            "defects" : [],
            "skipped" : []
        })

        for image_path in images_paths:
            # Load the image to be pasted
            image = Image.open(image_path)
            # Resize to desired size
            image = image.resize(desired_size, Image.LANCZOS)
            # Remove transparent parts by cropping the bounding box
            if image.mode == 'RGBA':
                bbox = image.getbbox()
                image = image.crop(bbox)

            # Extract the mask (alpha channel if present)
            mask = image.split()[-1] if image.mode == 'RGBA' else print(f"there is an error with: {image_path}")

            #resize image by a random factor (20% max)
            size_factor = random.uniform(*augmented_size_range)

            # Get the size of the image
            img_width, img_height = image.size
            img_width = int(img_width * size_factor)
            img_height = int(img_height * size_factor)

            # Resize image and mask with the random factor
            image = image.resize((img_width,img_height), Image.LANCZOS)
            mask = mask.resize((img_width,img_height), Image.LANCZOS)
            
            # for JSON annotation
            # Extract the label from the image path
            label = image_path.split("_")[1]

            # Add category id
            category_id = category_map[label]

            # DEFECTS AUGMENTATION SEGMENT

            # Randomly modify the brightness, but only slightly
            enhancer = ImageEnhance.Brightness(image)
            exposure_factor = random.uniform(*augmented_brightness_range)
            image = enhancer.enhance(exposure_factor)

            # Randomly rotate the image
            if random.random() < augmented_rotation_probability and (category_id != 2 and category_id != 4):
                # With a random angle
                rotate_angle = random.uniform(0,360)
                # Rotate both image and mask
                image = image.rotate(rotate_angle, expand=True)
                mask = mask.rotate(rotate_angle, expand=True)  
            
            # Randomly traspose the image (horizontally, vertically or both)
            if random.random() < augmented_transpose_probability:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < augmented_transpose_probability:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            # Update the dimensions of the resized image
            img_width, img_height = image.size

            # After all the transformation, check if the image needs to be resized to fit within the area
            if img_width > area_width or img_height > area_height:
                # If so, resize to fit
                scale = min(area_width / img_width, area_height / img_height)
                new_size = (int(img_width * scale), int(img_height * scale))
                image = image.resize(new_size, Image.LANCZOS)
                mask = mask.resize(new_size, Image.LANCZOS)

                # Update once again the dimensions of the resized image
                img_width, img_height = image.size

            # Find a random position that does not overlap with existing regions
            max_attempts = 100
            for _ in range(max_attempts):
                x_pos = random.randint(x_min, x_max - img_width)
                y_pos = random.randint(y_min, y_max - img_height)

                # Check for overlaps
                new_region = (x_pos, y_pos, x_pos + img_width, y_pos + img_height)

                if not any(
                    region[0] < new_region[2] and region[2] > new_region[0] and
                    region[1] < new_region[3] and region[3] > new_region[1]
                    for region in occupied_regions
                ):
                    occupied_regions.append(new_region)
                    break
            else:
                # In case you didn't find a spot to paste the image in, skip it
                # Add to log
                json_log['counters']['Skipped'] = json_log['counters']['Skipped'] + 1
                json_log['images'][image_id-1]['skipped'].append(os.path.basename(image_path))
                continue

            # Paste the image onto the background
            background.paste(image, (x_pos, y_pos), image if image.mode == 'RGBA' else None)

            # Paste the mask in the correct position
            combined_mask.paste(mask, (x_pos, y_pos))          

            # Calculate area
            area = img_width * img_height

            #add to counters log
            json_log['counters'][label] = json_log['counters'][label]+1
            json_log['images'][image_id - 1]['defects'].append(os.path.basename(image_path))

            # If the defect is not a component, add it to the json
            if category_id != 0:
                # Add annotation for the image
                json_data["annotations"].append({
                    "id": len(json_data["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_pos, y_pos, img_width, img_height],
                    "area": area,
                    "iscrowd": 0
                })
        
        # FOR loop end

        if len(json_log['images'][image_id-1]) == 0:
            json_log['counters']['Blank'] = json_log['counters']['Blank'] + 1

        # Save the combined mask
        combined_mask.save(mask_output_path)

        # Save the resulting image
        background.save(output_path)

        # Save the annotations to a JSON file
        with open(annotations_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        #save log
        with open(log_path, 'w') as json_file:
            json.dump(json_log, json_file, indent=4)

    except Exception as e:
        print("An error occurred while generating images")
        raise SystemExit(f"Error: {e}")

def generate_images(defects_path, dataset_name, output_base_path, background_path, model_path):
    """
    :param defects_path: path to defect images (default="Dataset\\Masks")
    :param dataset_name: name of dataset (default="defaultDataset")
    :param output_base_path: path to save the image in (default="Dataset\\DefectsImage\\defaultDataset")
    :param background_path: path to background images (default="Dataset\\NoDefects")
    :param model_path: path to model folder (default="HarmonizationModels") 
    """

    pattern_list = ["Component", "Hole", "Horizontal", "Spattering", "Vertical", "Incandescence"]
    # Find all images with one of the strings above in the name
    found_images = find_image_with_pattern(defects_path, pattern_list)

    # If some images found (check to avoid wrong folder)
    if found_images:
        pbar = tqdm(total=n_image_to_generate, desc="Generating images")

        #create json for annotations and log
        json_path = os.path.join(output_base_path, f"{dataset_name}.json")
        create_new_json(json_path)
        create_new_log_json(output_base_path)
        log_json_path = os.path.join(output_base_path,"log.json") 

        #set usable area to paste defects in
        usable_area = (245, 130, 1265, 880)

        # Set variables for selecting number of defects for each generated image
        defects_possible_numbers = [1, 2, 3, 4, 5, 6, 7, 8]
        defects_possible_weigths = [1, 2, 3, 4, 4, 3, 2, 1]
        # Set all to 1 to have uniform defects distribution
        pattern_weights = [1,1,1,1.5,1.5,1]
        # Use background path to generate image with different background
        # Get all models dir
        available_models = [dir for dir in os.listdir(model_path)]
        background_list =[]
        for model in available_models:
            # Append all background image paths to background list
            model_image = os.path.join(background_path,f"{model}.jpg")
            background_list.append(model_image)

        background_len = len(background_list)
        # Actually generate images
        for i in range(n_image_to_generate):
            pbar.update(1)
            # First, randomly selects which and how many defects to use for each image
            defects_number = random.choices(defects_possible_numbers, weights=defects_possible_weigths, k=1)[0]
            defect_category = random.choices(pattern_list, weights=pattern_weights, k=defects_number)
            # Then, select backgound image to use
            if background_len>1:
                background_number = random.randint(0, background_len-1)
            else:
                background_number = 0
            # If folder for this backgound image does not exist yet, create it
            if not os.path.exists(os.path.join(output_base_path,filename_no_extension(background_list[background_number]))):
                os.makedirs(os.path.join(output_base_path,filename_no_extension(background_list[background_number])))

            random_defects = []
            #for each defects category, randomly select the actual defect image
            for category in defect_category:
                random_defects.append(random.sample(found_images[category],k=1)[0])

            # Path and filename for each new generated image
            output_path = os.path.join(output_base_path,filename_no_extension(background_list[background_number]), f"{dataset_name}_output_{i+1}.png")
            # Path and filename for mask
            mask_output_path = os.path.join(output_base_path,filename_no_extension(background_list[background_number]), f"{dataset_name}_output_{i+1}_mask.png")
            
            # Generate one image with previously selected defects
            paste_multiple_images(background_list[background_number], random_defects, output_path, usable_area, mask_output_path,json_path,log_json_path)
    else:
        raise SystemExit(f"ERROR: check masks directory.\nNo masks found in {defects_path}")
    if pbar:
        pbar.close()

if __name__ == "__main__":
    # Identify project folder to locate the ConSinGan script
    script_path = os.path.abspath(__file__)  # Absolute path of the script
    current_dir = os.path.dirname(script_path)  # Actual last dir in absolute path
    
    src_path = None
    while current_dir:
        if os.path.basename(current_dir) == "src":
            src_path = current_dir
            break
        new_dir = os.path.dirname(current_dir)  # next folder
        if new_dir == current_dir:  # if root, abort
            break
        current_dir = new_dir
    if src_path is None:
        raise SystemExit(f"Unable to identify src path from {script_path}\nTry calling this script from src folder")

    parser = argparse.ArgumentParser(description="Generate images with defects")
    parser.add_argument("--dir", required=False, help="Path to dir with defects", default="Dataset\\Masks")
    parser.add_argument("--dataset_name", required=False, help="Name of the dataset to generate", default="defaultDataset")
    parser.add_argument("--background", required=False, help="Path for background images", default="Dataset\\NoDefects")
    parser.add_argument("--model", required=False, help="Path for models folder", default="HarmonizationModels")
    parser.add_argument("--output", required=False, help="Path for output images", default="Dataset\\DefectsImage")
    parser.add_argument("--n_images", required=False, help="Number of images to generate", default=10)

    args = parser.parse_args()

    n_image_to_generate = int(args.n_images)
    defects = ["Components", "Hole", "Horizontal", "Spattering", "Vertical", "Incandescence"]

    # Create the directory for the generated dataset
    image_directory = os.path.join(args.output, args.dataset_name)
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    # Generate images
    #folder_path, output_name, output_base_path,background_path, model_path
    generate_images(args.dir, args.dataset_name,image_directory, args.background,args.model) 

    # Harmonization script
    harmonization_script = f"py {src_path}\ConSinGan\evaluate_model.py"
    harmonization_args = "--gpu 0"
    model_dir =args.model

    used_harmonization_models = find_used_models(image_directory)
    # Build harmonization command
    i = 1
    #for each harmonization model actually used during generation
    for harmonization_model in used_harmonization_models:
        print(f"harmonizing model:{harmonization_model} [{i} of {len(used_harmonization_models)}]")
        i += 1
        actual_model_dir = os.path.join(model_dir,harmonization_model)
        # find all generated images to iterate on
        images_path = find_generated_images(os.path.join(image_directory,harmonization_model),args.dataset_name)
        command = f"{harmonization_script} {harmonization_args} --model_dir {actual_model_dir} --naive_dir {os.path.join(image_directory,harmonization_model)} --naive_dir_pattern {args.dataset_name}"
        
        result = subprocess.run(command, shell=True)
        if  result.returncode != 0:
            raise SystemExit(f"Error:{result}")
        # Iterate on all generated images
        for image_path in images_path:
            # image_path is: curent_path/datasetName/modelName/image.png
            # remove modelName from the path
            real_image_path = remove_model_from_path(image_path,harmonization_model)   
            # Double check if image does exits
            if os.path.isfile(image_path):
                # Copy harmonized image in output directory
                new_img_path = os.path.join(actual_model_dir, "Evaluation", os.path.basename((add_harmonized_to_filename(image_path))))
                shutil.move(new_img_path, real_image_path)
                
    print(f"Generation was successful.\nResult saved in {image_directory}\nCheck log in the same directory for further info about generated images")

