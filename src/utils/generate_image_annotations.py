from PIL import Image
import os
import argparse
import json

def create_new_json():
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
    
    return json_skeleton

def create_image_annotations(image_path, mask_list, json_data):
    category_map = { "Hole": 1, "Horizontal": 2, "Spattering": 3, "Vertical": 4, "Incandescence": 5 }

    image_name = os.path.basename(image_path)
    image = Image.open(image_path)
    img_width , img_height = image.size

    # Add image information to JSON file
    image_id = len(json_data["images"]) + 1
    json_data["images"].append({
        "id": image_id,
        "file_name": image_name,
        "width": img_width,
        "height": img_height
    })

    for mask in mask_list:
        img_mask = Image.open(mask)

        if img_mask.mode == 'RGBA':
            img_mask_area = img_mask.getbbox()

        x_min, y_min, x_max, y_max = img_mask_area
        img_width = x_max - x_min
        img_height = y_max - y_min

        area = img_width * img_height

        category_name = os.path.basename(mask).split("_")[1]

        # Add annotation for the image
        json_data["annotations"].append({
            "id": len(json_data["annotations"]) + 1,
            "image_id": image_id,
            "category_id": category_map[category_name],
            "bbox": [x_min, y_min, img_width, img_height],
            "area": area,
            "iscrowd": 0
        })

def filter_image_path(entry, image_name):
    if entry.is_file():
        entry_defect = entry.name.split("_")[1]

        if entry_defect == "Component":
            return False       
           
        return entry.name.split("_")[0] == image_name
    return False
          
def find_mask_path(image_name, defects_dir):
    masks_list = list(
        map(
            lambda entry: entry.path, 
            filter(
                lambda entry: filter_image_path(entry,image_name),
                os.scandir(defects_dir)
            )
        )
    )
    return masks_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates annotations for real images.")

    parser.add_argument("--masks_path",
                        type=str,
                        help="Defect masks path.", 
                        default="../../Dataset/Masks/Test")
    parser.add_argument("--images_path",
                        type=str,
                        help="Real images path.", 
                        default="../../Dataset/Defects/Test")
    parser.add_argument("--annotations_path",
                        type=str,
                        help="Annotations json file path.", 
                        default="../Data/annotations/test_annotations.json")

    args = parser.parse_args()

    if os.path.exists(args.annotations_path):
        with open(args.annotations_path, 'r') as json_file:
            json_data = json.load(json_file)
    else:
        json_data = create_new_json()

    for entry in os.scandir(args.images_path):
        file_name = entry.name.split(".")[0]
        defects_list = find_mask_path(file_name, args.masks_path)
        create_image_annotations(entry.path, defects_list, json_data)
    
    with open(args.annotations_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Annotations successfully saved to {args.annotations_path}")