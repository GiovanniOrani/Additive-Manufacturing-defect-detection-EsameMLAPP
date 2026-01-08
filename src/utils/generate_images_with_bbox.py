from PIL import Image, ImageDraw, ImageFont
import argparse
import json
import os

def draw_bounding_boxes(images_path, annotations_path, output_path=None):
    category_map = {
        1: ("Hole", "red"), 
        2: ("Horizontal", "green"), 
        3: ("Spattering", "blue"), 
        4: ("Vertical", "yellow"), 
        5: ("Incandescence", "magenta")
    }

    with open(annotations_path,"r") as json_file:
        annotations = json.load(json_file)

    for image_entry in os.scandir(images_path):
        image_path = image_entry.path
        image_name = os.path.basename(image_path).replace("_harmonized", "")
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        #Search for the image ID in the annotations
        for img in annotations["images"]:
            if img["file_name"] == image_name:
                image_id = img["id"]
        
        # Draw each bounding box
        for annotation in annotations["annotations"]:
            if annotation["image_id"] != image_id:
                continue

            bbox = annotation["bbox"]
            x, y, w, h = bbox
            margin = 5
            
            x = x-margin
            y = y-margin
            w = w + (margin * 2)
            h = h + (margin * 2)

            # Define the rectangle's corners
            rect = [x, y, x + w, y + h]
            
            # Add the category ID or label
            category_id = (category_map[annotation["category_id"]])[0]
            category_color = category_map[annotation["category_id"]][1]

            # Draw the rectangle (bounding box)
            draw.rectangle(rect, outline=category_color, width=1)

            font = ImageFont.truetype("arial.ttf", 10)
            draw.text((x, y - 12), f"{category_id}", fill=category_color, font=font)
        
        # Save or display the image
        if output_path:
            image.save(os.path.join(output_path, image_name))
            print(f"Image with bounding boxes saved to {output_path}")
        else:
            image.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates images with bounding boxes.")

    parser.add_argument("--images_path",
                        type=str,
                        required=True, 
                        help="Images path to process.")
    parser.add_argument("--annotations_path",
                        type=str,
                        required=True, 
                        help="Annotations json file path.")
    parser.add_argument("--output_path",
                        type=str,
                        required=True, 
                        help="Path in which store generated images.")

    args = parser.parse_args()

    draw_bounding_boxes(args.images_path, args.annotations_path, args.output_path)