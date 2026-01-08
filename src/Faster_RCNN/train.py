import torch.optim as optim
import torch
from Models import faster_RCNN
from Datasets import my_dataset
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as T
import torchvision
import torchvision.ops as ops
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from utils import poly_lr_scheduler

category_map = {
    1: ("Hole", "red"), 
    2: ("Horizontal", "green"), 
    3: ("Spattering", "blue"), 
    4: ("Vertical", "yellow"), 
    5: ("Incandescence", "magenta")
}

def intersection_percentage(box1, box2) -> float:
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    inter_width = max(0, min(x2, x2_p) - max(x1, x1_p))
    inter_height = max(0, min(y2, y2_p) - max(y1, y1_p))
    intersection = inter_width * inter_height
    
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x2_p - x1_p) * (y2_p - y1_p)
    
    return ((intersection / area_box1) + (intersection / area_box2))/2  if area_box1 > 0 and area_box2 > 0 else 0.0

def nms_per_category(boxes,category,scores,threshold = 0.5):
    filtered_boxes = []
    filtered_categories = []
    filtered_scores = []
    unique_categories = category.unique()
    for cat in unique_categories:
        mask = category == cat
        cat_boxes = boxes[mask]
        cat_scores = scores[mask]
        keep = ops.nms(cat_boxes, cat_scores, threshold)
        filtered_boxes.append(cat_boxes[keep])
        filtered_categories.append(torch.full((len(keep),), cat))
        filtered_scores.append(cat_scores[keep])

    if filtered_boxes:
        filtered_boxes = torch.cat(filtered_boxes, dim=0)
        filtered_categories = torch.cat(filtered_categories, dim=0)
        filtered_scores = torch.cat(filtered_scores, dim=0)
    else:
        filtered_boxes = torch.empty((0, 4))
        filtered_categories = torch.empty((0,))
        filtered_scores = torch.empty((0,))
    return filtered_boxes, filtered_categories, filtered_scores

def drawbbox(image,boxes,category,scores,threshold,iou_remove_threshold,filename_w_path=None):
    # Convert image from Tensor to PIL
    image_pil = T.ToPILImage()(image.cpu())

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image_pil)

    #remove box with score less than threshold
    filtered_boxes = nms_per_category(boxes.cpu(),category.cpu(),scores.cpu(),iou_remove_threshold)
    filtered_boxes = list(zip(filtered_boxes[0].numpy(),filtered_boxes[1].numpy(),filtered_boxes[2].numpy()))

    for box, cat, score in filtered_boxes:
        if score < threshold:
            # Don't draw any box with not enough confidence
            continue
        x_min, y_min, x_max, y_max = box
        rect_coord = [x_min, y_min, x_max - x_min, y_max - y_min]

        # Add the category ID or label
        category_id = category_map[cat][0]
        category_color = category_map[cat][1]

        # Draw the bounding box
        x, y, w, h = rect_coord
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=category_color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, category_id, color=category_color, fontsize=8)

    plt.axis("off")

    if filename_w_path is None:
        plt.show()
    else:
        plt.savefig(filename_w_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_result_images(model, dataloader, threshold, path_to_save=None, iou_remove_threshold = 0.5):
    # Put model in evaluation mode
    model.eval()
    i = 1

    for images, targets in dataloader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        next_image = 0
        #compute predicted label for each image in dataloader
        outputs = model(images)
        for output in outputs:
            filename = None

            if path_to_save is not None:  
                filename = os.path.join(path_to_save,f"result_{i}.png")

            i+=1
            pred_boxes = output["boxes"].detach().cpu()
            pred_labels = output["labels"].detach().cpu()
            scores = output["scores"].detach().cpu()

            drawbbox(images[next_image], pred_boxes, pred_labels, scores, threshold,iou_remove_threshold, filename_w_path=filename )
            next_image+=1

    if path_to_save is not None:
        print(f"results images with box saved in {path_to_save}")


def calculate_miou(pred_boxes, true_boxes):
    # Compute mIoU for all prediction boxes with the true box that better match
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return 0

    iou_matrix = torchvision.ops.box_iou(pred_boxes, true_boxes)
    max_ious, _= iou_matrix.max(dim=1)

    return max_ious.mean().item()

def ap_trapezoidal(recalls, precisions):
    # Compute AP with trapezoidal rule for estimating the area under the precision-recall curve

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # Add start point (0,1) and end point (1,0) if not yet present in the p-r curve
    if recalls[0] > 0:
        recalls = np.insert(recalls, 0, 0.0)
        precisions = np.insert(precisions, 0, 1.0)

    if recalls[-1] < 1:
        recalls = np.append(recalls, 1.0)
        precisions = np.append(precisions, 0.0)
    
    # compute area under p-r curve
    ap = np.trapz(precisions, recalls)

    return ap

def calculate_standard_ap(recalls, precisions):
    # Compute AP in the standard way

    #already sorted by recall and score, add start and end point if needed
    if recalls[0] > 0:
        recalls = np.insert(recalls, 0, 0.0)
        precisions = np.insert(precisions, 0, 1.0)

    if recalls[-1] < 1:
        recalls = np.append(recalls, 1.0)
        precisions = np.append(precisions, 0.0)

    recalls = torch.tensor(recalls)
    precisions = torch.tensor(precisions)
    # Make sure precision values are monotonic decreasing 
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    indices = torch.where(recalls[1:] != recalls[:-1])[0] + 1
    # Take only points where recall changes to compute the AP
    ap = torch.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap.item()

def validate(model, dataloader, device, iou_threshold=0.5):
    # Validate model performance
    print('Start Validation')
    # Put model in evaluation mode
    model.eval()

    miou_per_class = {}
    ap_per_class = {}
    class_counts = {}
    p_per_class = {}
    f1_per_class = {}

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Get predicted label for each image in the dataset
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes =  output["boxes"].detach().cpu()
                pred_labels = output["labels"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()
                true_boxes = targets[i]["boxes"].cpu()
                true_labels = targets[i]["labels"].cpu()

                
                if len(pred_boxes) == 0 and len(true_boxes)==0:
                    # If no box at all, skip
                    continue

                # For each class
                for cls in range(1, 6):
                    # Keep only class related object
                    pred_mask = pred_labels == cls
                    true_mask = true_labels == cls
                    pred_cls_scores = pred_scores[pred_mask]
                    pred_cls_boxes = pred_boxes[pred_mask]
                    true_cls_boxes = true_boxes[true_mask]

                    # If no predicted boxes or true boxes 
                    if len(pred_cls_boxes) == 0 or len(true_cls_boxes) == 0:
                        # If no match in prediction - true
                        if len(pred_cls_boxes) != len(true_cls_boxes)!=0:
                            miou_per_class.setdefault(cls, []).append(0)
                            ap_per_class.setdefault(cls, []).append(0)
                            p_per_class.setdefault(cls, []).append(0)
                            f1_per_class.setdefault(cls, []).append(0)
                        # Skip
                        continue

                    miou = calculate_miou(pred_cls_boxes, true_cls_boxes)
                    miou_per_class.setdefault(cls, []).append(miou)

                    class_counts[cls] = class_counts.get(cls, 0) + 1
                    

                    #initialize value
                    tp=0
                    fp=0
                    fn=len(true_cls_boxes)
                    detected_prediction = np.full(len(true_cls_boxes), False, dtype=bool) 
                    precision=[]
                    recall=[]
                    # Get IoU matrix for class boxes
                    intersection_matrix = torchvision.ops.box_iou(pred_cls_boxes, true_cls_boxes)
                    # For each predicted boxes
                    for i in range(len(pred_cls_boxes)):
                        # Get max IoU
                        iou_scores = intersection_matrix[i]
                        max_iou, max_iou_idx = iou_scores.max(0)
                        # If box does match with a true box
                        if max_iou >= iou_threshold:
                            tp+=1 # True positive + 1
                            # If the box predicted was not already found
                            if not detected_prediction[max_iou_idx]:
                                detected_prediction[max_iou_idx]=True
                                fn-=1 # False negative - 1
                        else:
                            # Box is not an accurate prediction
                            fp+=1 # False positive + 1
                        # compute precision and recall
                        precision.append( tp / (tp + fp) if (tp + fp) > 0 else 1.0 )
                        recall.append( tp / (tp + fn) if (tp + fn) > 0 else 0.0 )
                    
                    # If at least one prediction done
                    if(len(precision)>0):
                        # Get total predicion and recall
                        score_precision=precision[-1]
                        score_recall = recall[-1]
                    else:
                        score_precision=0
                        score_recall=0
                    # Use trapezoidal AP because is more accurate in our case
                    ap = ap_trapezoidal(recall,precision)
                    #ap = calculate_standard_ap(recall, precision)
                    ap_per_class.setdefault(cls, []).append(ap)
                    p_per_class.setdefault(cls, []).append(precision[-1])
                    f1_score = 2 *((score_precision*score_recall)/(score_precision+score_recall)) if (score_precision+score_recall)>0 else 0.0
                    f1_per_class.setdefault(cls, []).append( f1_score )

    for cls in range(1, 6): # If you don't have any instances of a class in the dataset
        if cls not in ap_per_class:
            ap_per_class.setdefault(cls, []).append(-1)
            miou_per_class.setdefault(cls, []).append(-1)
            p_per_class.setdefault(cls, []).append(-1)
            f1_per_class.setdefault(cls, []).append(-1)
    # Compute all aggregate metrics
    mean_miou_per_class = {cls: sum(values) / len(values) for cls, values in miou_per_class.items()}
    mean_ap_per_class = {cls: sum(values) / len(values) for cls, values in ap_per_class.items()}
    mean_precision_per_class = {cls: sum(values) / len(values) for cls, values in p_per_class.items()}
    mean_f1_per_class = {cls: sum(values) / len(values) for cls, values in f1_per_class.items()}
    mean_iou = sum(mean_miou_per_class.values()) / len(mean_miou_per_class) if mean_miou_per_class else 0
    mean_ap = sum(mean_ap_per_class.values()) / len(mean_ap_per_class) if mean_ap_per_class else 0

    return mean_iou, mean_ap, mean_miou_per_class, mean_ap_per_class, mean_precision_per_class, mean_f1_per_class

def train(args, device, model, optimizer, train_dataloader, val_dataloader):
    print("Start Training")
    best_score = 0
    loss_history = []
    best_model_path = None 
    for epoch in range(args.num_epochs):
        # Change LR based on polinomial
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        # Put model in train mode
        model.train()
        tq = tqdm(total=len(train_dataloader) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))

        loss_record = []
        # Do a full epoch of training
        for images, targets in train_dataloader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % losses)
            loss_record.append(losses.item())
            loss_history.append(losses.item())
        tq.close()

        loss_train_mean = np.mean(loss_record)
        print('Loss for train : %f' % loss_train_mean)

        # Validation
        mean_iou, mean_ap, miou_per_class, ap_per_class,p_per_class,f1_per_class = validate(model, val_dataloader, device)
        mean_f1 = sum(f1_per_class.values()) / len(f1_per_class) if f1_per_class else 0
        # Print important information for each epoch
        print('Validation results: mIoU = %.6f mAP = %.6f mF1 = %.6f' % (mean_iou, mean_ap,mean_f1))
        print("\nIoU per class:")
        for cls, miou in miou_per_class.items():
            print(f"{category_map[cls][0]}: IoU = {miou:.4f}")

        print("\nAP per class:")
        for cls, ap in ap_per_class.items():
            print(f"{category_map[cls][0]}: AP = {ap:.4f} | F1 = {f1_per_class[cls]:.4f}")

        # Saving model weights
        score = 0.7 * mean_ap + 0.3 * mean_iou
        #score = 0.6 * mean_ap + 0.3 * mean_iou + 0.1 * mean_f1
        # Save model only if the score is better than the last saved model
        if score > best_score:
            best_score = score

            # Delete the old model if exist
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)

            best_model_path = os.path.join(os.path.dirname(args.save_model_path), f"best_model_{epoch}.pth")
            torch.save(model.state_dict(), best_model_path)
    
    # Plot overall loss after training
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over All Epochs')
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(args.save_model_path), 'loss_plot.png'))  # Save the final plot
    plt.show()   

def parse_args():
    parse = argparse.ArgumentParser()
    
    parse.add_argument('--root',
                       dest='root',
                       type=str,
                       default='../Data')
    parse.add_argument('--num_epochs',
                       type=int, 
                       default=50,
                       help='Number of epochs to train for')
    parse.add_argument('--batch_size',
                       type=int,
                       default=2,
                       help='Number of images in each batch')
    parse.add_argument('--learning_rate',
                        type=float,
                        default=0.005,
                        help='learning rate used for train')
    parse.add_argument('--num_classes',
                       type=int,
                       default=6,
                       help='num of object classes (with void)')
    parse.add_argument('--use_gpu',
                       type=bool,
                       default=True,
                       help='whether to user gpu for training')
    parse.add_argument('--save_model_path',
                       type=str,
                       default='./Trained_Models',
                       help='path to save model')
    parse.add_argument('--momentum',
                       type=float,
                       default=0.9,
                       help='Momentum component of the optimiser')
    parse.add_argument('--weight_decay',
                       type=float,
                       default=5e-4,
                       help='Regularisation parameter for L2-loss')
    parse.add_argument('--augmentation',
                       type=bool,
                       default=False,
                       help='whether to apply Data Augmentation for training')
    parse.add_argument('--mode',
                       type=str,
                       default='train',
                       help='mode of execution (train or test)')
    parse.add_argument('--filter_iou',
                       type=float,
                       default=0.5,
                       help='define how much one box need to overlap with another higher score box to be filter out')
    
    return parse.parse_args()

if __name__ == '__main__':
    args = parse_args()

    root = args.root
    num_classes = args.num_classes
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model = faster_RCNN.get_model(num_classes)
    model = model.to(device)

    if args.mode == 'train': 
        if not os.path.exists(args.save_model_path):
            # If saved model is a file (loading a existing model) but doesn't exist
            if '.' in os.path.basename(args.save_model_path):
                assert 0, 'save_model_path is a file but it does not exist!'
            else:
            # Else is a dir that doesn't exist, create it
                os.makedirs(args.save_model_path)
        else:
            # If saved model is a file
            if os.path.isfile(args.save_model_path):
                # Load it
                model.load_state_dict(torch.load(args.save_model_path, map_location=device))

        train_dataset_full =my_dataset.CustomDataset(root, augmentation=args.augmentation)

        # Splitting training dataset in 80% training, 20% validation
        indexes = range(0, len(train_dataset_full))
        splitting = train_test_split(indexes, train_size = 0.8, random_state = 42, shuffle = True)
        train_indexes = splitting[0]
        val_indexes = splitting[1]

        train_dataset = Subset(train_dataset_full, train_indexes)
        val_dataset = Subset(train_dataset_full, val_indexes)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

        train(args, device, model, optimizer, train_dataloader, val_dataloader)
    else:
        assert args.save_model_path is not None, 'save_model_path parameter must be defined.'
        if not os.path.exists(args.save_model_path):
            assert 0, 'save_model_path parameter is not an existing file'
        model.load_state_dict(torch.load(args.save_model_path, map_location=device))

        test_dataset =my_dataset.CustomDataset(root, split=args.mode)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

        print("Testing")
        mean_iou, mean_ap, miou_per_class, ap_per_class, precision_per_class, f1_per_class = validate(model, test_dataloader, device)
        print('Results: mIoU = %.4f mAP = %.4f mF1=%.4f' % (mean_iou, mean_ap,sum(f1_per_class.values()) / len(f1_per_class) if f1_per_class else 0))
        print("\nIoU per class:")
        for cls, miou in dict(sorted(miou_per_class.items())).items():
            print(f"{category_map[cls][0]}: IoU = {miou:.4f}")

        print("\nAP per class:")
        for cls, ap in dict(sorted(ap_per_class.items())).items():
            print(f"{category_map[cls][0]}: AP = {ap:.4f} | P:{precision_per_class[cls]} - F1:{f1_per_class[cls]}")

        if not os.path.exists("result"):
            os.makedirs("result")
        save_result_images(model, test_dataloader, 0.5, "result",iou_remove_threshold=args.filter_iou)