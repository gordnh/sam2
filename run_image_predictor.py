from datetime import datetime
import os
from typing import List
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageFile

DEVICE_TYPE_OVERRIDE = 'cpu'

## Prevent plt from rendering other artifacts when saving images, such as axes, titles etc
IMAGES_PRESERVE_SIZE = True

BINARY_MASK = True

SHOW_IMAGES = False

print(f"WARN: {DEVICE_TYPE_OVERRIDE=}")

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if DEVICE_TYPE_OVERRIDE:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if BINARY_MASK:
        color = np.array([0/255, 0/255, 0/255, 1.0])
    elif random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, image_width, image_height, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # plt.figure(figsize=(10, 10))
        dpi = 80
        plt.figure(figsize=(image_width/dpi, image_height/dpi))
        if not BINARY_MASK:
            plt.imshow(image)
        show_mask(mask, plt.gca(), borders=(BINARY_MASK and borders))
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1 and not IMAGES_PRESERVE_SIZE:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        if SHOW_IMAGES:
            plt.show()
        else:
            relative_path = f'output_images/show_masks/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
            plt.savefig(relative_path)
            print(f"Saved mask at: {os.path.abspath(relative_path)}")

def save_masked_object_as_png(image, mask, output_path='masked_object.png'):
    """
    Saves the masked object from an image as a PNG.
    
    Args:
        image (ndarray): The original image array.
        mask (ndarray): The mask array with the same width and height as the image.
        output_path (str): Path where the output PNG should be saved.
    """
    # Ensure mask is uint8 to match standard image formats
    mask_image = (mask.astype(np.uint8) * 255)

    # Create an empty alpha channel image
    alpha_channel = Image.fromarray(mask_image, mode='L')

    # Convert the original image array to an Image object
    original_img = Image.fromarray(image)

    # Convert the image to RGBA if it's not already
    rgba_img = original_img.convert("RGBA")

    # Place the alpha channel as the new alpha layer of the image
    rgba_img.putalpha(alpha_channel)

    # Save the resulting image with alpha transparency to keep only the masked object visible
    rgba_img.save(output_path, "PNG")
    if SHOW_IMAGES:
        # Display the final image with matplotlib
        plt.imshow(rgba_img)
        plt.axis('off')  # Hide axis
        plt.title("Masked Object")
        plt.show()
        
    print(f"Saved masked object as PNG at: {output_path}")

# image = Image.open(f'{Path.home()}/canva-ai-hackathon/hot-air-balloon.jpg')

def apply_sam2_mask(image: ImageFile, coords: List[List[int]], labels: List[int]):

    image_width, image_height = image.size
    image = np.array(image.convert("RGB"))

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    if SHOW_IMAGES:
        plt.show()
    else:
        init_image_path = f'output_images/init_image/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        plt.savefig(init_image_path)
        print(f'Saved initial image at: {init_image_path}')


    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)

    predictor.set_image(image)

    # input_point = np.array([[image_width/2+120, image_height/2]])
    input_point = np.array(coords)
    input_label = np.array(labels)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    if SHOW_IMAGES:
        plt.show()
    else:
        init_image_with_points_path = f'output_images/init_image_with_points/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        plt.savefig(init_image_with_points_path)
        print(f"Saved initial image with points at: {init_image_with_points_path}")

    print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    masks.shape  # (number_of_masks) x H x W

    show_masks(image, image_width, image_height, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

    masked_image_relative_path = f'output_images/masked_image/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.png'
    save_masked_object_as_png(image, masks[0], output_path=masked_image_relative_path)

    masked_image = Image.open(f'{Path.home()}/canva-ai-hackathon/sam2-fork/{masked_image_relative_path}')

    print("Finished")
    return masked_image


if __name__ == '__main__':
    pass