# -----------------------------------------
# Spatial Video Converter 1.0
# Scooterrr 2024
# https://bsky.app/profile/supscooterr.com
# -----------------------------------------
# Takes a directory of images and generates a spatial video from them for viewing natively on Apple Vision Pro and Meta HMDs.
# The images are processed via a MiDaS model to generate a depth map, which is then used to displace the pixels in the image.
# The displaced images are then converted to a mv-hevc spatial video using the spatial CLI tool from Mike Swanson https://blog.mikeswanson.com/spatial.

# Requirements:
# - OpenCV, PyTorch, ImageIO, FFMPEG


import cv2
import torch
import argparse
import numpy as np
import os
import imageio.v2 as imageio

class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


print(TerminalColors.HEADER +"""
███████╗ ██████╗ ██████╗  ██████╗ ████████╗██████╗ ███████╗██╗   ██╗
██╔════╝██╔════╝██╔═══██╗██╔═══██╗╚══██╔══╝██╔══██╗██╔════╝██║   ██║
███████╗██║     ██║   ██║██║   ██║   ██║   ██║  ██║█████╗  ██║   ██║
╚════██║██║     ██║   ██║██║   ██║   ██║   ██║  ██║██╔══╝  ╚██╗ ██╔╝
███████║╚██████╗╚██████╔╝╚██████╔╝   ██║   ██████╔╝███████╗ ╚████╔╝ 
╚══════╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝   ╚═════╝ ╚══════╝  ╚═══╝                                                                                             
Spatial Video Converter 1.0
""" + TerminalColors.ENDC)

# Takes an input image and generates a depth map using a MiDaS model from the pytorch hub
def generate_depth_map(input_image_path):
    print(TerminalColors.OKBLUE + "[Depth]: --------------------" + TerminalColors.ENDC)
    print("[Depth]: Generating depth map from {path}...".format(path=input_image_path))
    
    # Download a model from the pytorch hub
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)    print("[Depth]: Loading model: ", model_type)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("[Depth]: Moving model to: ", device)
    midas.to(device)
    midas.eval()

    # Load transforms to resize and normalize the image for large or small model
    print("[Depth]: Loading transforms...")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Load the image and apply tranforms
    print("[Depth]: Loading and transforming image...")
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    # Predict and resize to original resolution
    print("[Depth]: Predicting and resizing...")
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    # Apply level adjustment to increase value of bright areas on output depth map
    print("[Depth]: Applying level adjustment...")
    output = cv2.normalize(output, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output = cv2.equalizeHist(output)

    # Return result
    print(TerminalColors.OKGREEN + "[Depth]: Depth map generated." + TerminalColors.ENDC)
    return output

# Displaces pixels on an image by a specified amount along the X axis.
def apply_displacement_map(src_img, disp_map, disp_amount, eye='unspecified'):
    print(f"[Displace]: Applying displacement map for {eye} eye...")
    rows, cols = src_img.shape[:2]
    map_x = np.tile(np.arange(cols), (rows, 1)).astype(np.float32)
    map_y = np.repeat(np.arange(rows), cols).reshape((rows, cols)).astype(np.float32)
    
    # Adjust maps according to displacement map
    map_x += disp_map * disp_amount
    
    displaced_img = cv2.remap(src_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return displaced_img

# Exports a .mp4 from SBS image ( This is just until I figure out how to author stereo HEIC )
def create_video_from_image(image_path, output_video_path, duration=1, fps=30):
    kargs = { 'macro_block_size': None }
    writer = imageio.get_writer(output_video_path, fps=fps, **kargs)
    
    image = imageio.imread(image_path)
    total_frames = duration * fps
    
    for _ in range(total_frames):
        writer.append_data(image)
    
    writer.close()

# Argument parser
parser = argparse.ArgumentParser(description='This script processes an input directory of images, converting each to a Spatial Video ( mv-hevc ).')
parser.add_argument('-i', '--input', help='Path to the input directory. All PNG and JPEG files in this directory will be converted to Spatial Videos.', required=True)
parser.add_argument('-s', '--separation', type=int, help='Separation value, how much distortion is applied to the image, smaller value makes things feel bigger.', required=True)
parser.add_argument('--sbs', action='store_true', help='Export the side by side image and video.')
parser.add_argument('--depth', action='store_true', help='Export the depth map.')
parser.add_argument('-o', '--output-directory', help='Path to the output directory. The spatial videos will be saved in this directory.', default=None)
args = parser.parse_args()

# Get list of images in the input directory
input_images = [img for img in os.listdir(args.input) if img.endswith(('.jpeg', '.png', '.jpg'))]

print(TerminalColors.WARNING + f"[Info]: Found {len(input_images)} images in {args.input}.")
print(f"[Info]: Beginning conversion with separation value of {args.separation}..." + TerminalColors.ENDC)

for image in input_images:
    # Generate output file names from input file name
    input_filename = os.path.splitext(os.path.basename(image))[0]
    output_image_path = input_filename + '_sbs.png'
    output_video_path = input_filename + '_sbs.mp4'
    output_spatial_video_path = os.path.join(args.output_directory if args.output_directory else args.input, input_filename + '_spatial.mov')
    output_depth_map_path = input_filename + '_depth.png'

    # Generate depth map
    depth_map = generate_depth_map(os.path.join(args.input, image))

    # Normalize depth map values between -1 and 1
    depth_map = (depth_map / 255.0) * 2.0 - 1.0

    # Create displaced image for both eyes
    print(TerminalColors.OKBLUE + "[Displace]: --------------------" + TerminalColors.ENDC)
    src_img = cv2.imread(os.path.join(args.input, image))
    left_eye = apply_displacement_map(src_img, depth_map, -args.separation, 'left')
    right_eye = apply_displacement_map(src_img, depth_map, args.separation, 'right')

    # Create a side by side image of both eyes
    print(TerminalColors.OKBLUE + "[Composite]: --------------------" + TerminalColors.ENDC)
    print("[Composite]: Creating side by side image...")
    displaced_img = np.concatenate((left_eye, right_eye), axis=1)
    print(TerminalColors.OKGREEN + "[Composite]: Composite generated." + TerminalColors.ENDC)

   # Save the depth map
    print(TerminalColors.OKBLUE + "[Output]: --------------------" + TerminalColors.ENDC)
    print("[Output]: Saving depth map...")
    cv2.imwrite(output_depth_map_path, depth_map)

    # Save the displaced image
    print("[Output]: Saving displaced image...")
    cv2.imwrite(output_image_path, displaced_img)

    # Create a video from the displaced image
    print("[Output]: Creating video from displaced image...")
    create_video_from_image(output_image_path, output_video_path)

    # Convert the video to a mv-hevc spatial video
    print("[Output]: Converting video to mv-hevc spatial video...")
    print(f"[Output]: './spatial make -i {output_video_path} -f sbs -o {output_spatial_video_path} --cdist 60 --hfov 60 --hadjust 0.02 --projection rect --bitrate 60M --use-gpu'")
    os.system(f'./spatial make -i {output_video_path} -f sbs -o {output_spatial_video_path} --cdist 60 --hfov 60 --hadjust 0.02 --projection rect --bitrate 60M --use-gpu')

    print(TerminalColors.OKGREEN + "[Output]: Outputs complete." + TerminalColors.ENDC)

    # Clean up the extra files, unless requested.
    if not args.sbs:
        os.remove(output_image_path)
        os.remove(output_video_path)
    if not args.depth:
        os.remove(output_depth_map_path)

print(TerminalColors.OKGREEN + "[:D]: All done!" + TerminalColors.ENDC)
