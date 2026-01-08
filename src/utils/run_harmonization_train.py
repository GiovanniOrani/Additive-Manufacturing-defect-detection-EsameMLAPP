import os
import shutil

def get_most_recent_folder(directory):
    # Get all dir in folder
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    # find most recent one
    most_recent_folder = max(folders, key=lambda folder: os.path.getmtime(os.path.join(directory, folder)))
    
    return os.path.join(directory,most_recent_folder)

def move_files(src_directory, dest_directory):
    # Check if the destination folder exists, if not, create it
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    
    # Get all items (files and folders) in the source directory
    for item in os.listdir(src_directory):
        src_path = os.path.join(src_directory, item)
        
        # Check if the item is a file (not a folder)
        if os.path.isfile(src_path):
            dest_path = os.path.join(dest_directory, item)
            
            # Move the file to the destination folder
            shutil.move(src_path, dest_path)

if __name__ == "__main__":
    # Identify project folder to locate the ConSinGan script
    script_path = os.path.abspath(__file__)  # Absolute path of the script
    execution_path = os.getcwd() #folder of execution
    current_dir = os.path.dirname(script_path)  # Actual last dir in absolute path
    root_path = None
    while current_dir:
        if os.path.basename(current_dir) == "mla-prj-23-mla-am04-g2":
            root_path = current_dir
            break
        new_dir = os.path.dirname(current_dir)  # next folder
        if new_dir == current_dir:  # if root, abort
            break
        current_dir = new_dir
    if root_path is None:
        raise SystemExit(f"Unable to identify project root path from {script_path}\nWas the file moved outside of project folder?")

    #get the relative path from the root path to the execution folder
    relative_root_path = os.path.relpath( root_path , execution_path)

    # Image ID: put here the images without defects that you want to train the model on
    image_ids = [ ] # put here the image number you want to train on, like this [1, 2 , 5 , 11 , .... ]
    if len(image_ids)==0:
        raise SystemExit("Script execution aborted.\nPut the image numbers you want to train on in the image ids list before running this script!")
    
    # Get path for noDefects images
    image_base_path = os.path.join(relative_root_path,"Dataset","NoDefects")
    # Get path for training script and model
    train_path = os.path.join(relative_root_path,"src","ConSinGAN","main_train.py")
    harmonization_models_path = os.path.join(relative_root_path,"src","ConSinGAN","HarmonizationModels")
    trained_models_path = os.path.join(relative_root_path,"Dataset","NoDefects")

    # Command's parameters
    command_template = (
        "python {} --gpu 0 --train_mode harmonization "
        "--train_stages 2 --min_size 512 --max_size 640 "
        "--lrelu_alpha 0.3 --niter 1000 --batch_norm --input_name {}"
    )

    # Run command for each image
    for img_id in image_ids:
        #image path is the image to train on
        image_path = os.path.join(image_base_path,f"Image{img_id}.jpg")

        #run the training script
        command = command_template.format(train_path,image_path)
        print(f"Running: {command}")
        os.system(command)

        #trained_model_path is the directory on which the ConSinGan training script put the trained model
        trained_model_path = os.path.join(relative_root_path,"TrainedModels","Dataset","NoDefects",f"Image{img_id}")
        #harmonization_model_path is the directory used by image_generation script
        harmonization_model_path =os.path.join(harmonization_models_path,f"Image{img_id}")
        #get the most recent folder in trained_model_path (correspond to the model you just trained)
        trained_model_folder = get_most_recent_folder(trained_model_path)

        #move trained model to the correct folder used by image_generation script
        print(f"Moving model for image_generation.py script\nFrom {trained_model_folder} to {harmonization_model_path}")
        move_files(trained_model_folder,harmonization_model_path)