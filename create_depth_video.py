
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
import os
from depth_anything_3.api import DepthAnything3

def depth_to_grayscale(depth, percentile=2):
    """
    Convert a depth map to a grayscale image.
    """
    depth = depth.copy()
    valid_mask = depth > 0
    
    if valid_mask.sum() > 10:
        depth_min = np.percentile(depth[valid_mask], percentile)
        depth_max = np.percentile(depth[valid_mask], 100 - percentile)
    else:
        depth_min = 0
        depth_max = 1.0
        
    if depth_min == depth_max:
        depth_min -= 1e-6
        depth_max += 1e-6

    # Normalize to [0, 1] and invert
    depth[valid_mask] = (depth_max - depth[valid_mask]) / (depth_max - depth_min)
    depth[valid_mask] = depth[valid_mask].clip(0, 1)
    
    # Set invalid depth to 0
    depth[~valid_mask] = 0
    
    # Scale to [0, 255] and convert to uint8
    grayscale_img = (depth * 255).astype(np.uint8)
    
    return grayscale_img

def process_video(video_input, video_output, model, process_res, progress_callback=None, stop_event=None):
    """
    Processes a single video file to create a depth map video.
    The model is passed as an argument to avoid reloading it for each video.
    """
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_input}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, fps, (frame_width, frame_height), isColor=True)

    try:
        for frame_idx in range(total_frames):
            if stop_event and stop_event.is_set():
                print("Processing stopped.")
                break
            
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            prediction = model.inference(
                [frame_rgb],
                process_res=process_res
            )
            
            depth_map = prediction.depth[0]
            grayscale_depth = depth_to_grayscale(depth_map)
            output_frame = cv2.cvtColor(grayscale_depth, cv2.COLOR_GRAY2BGR)
            output_frame = cv2.resize(output_frame, (frame_width, frame_height))

            out.write(output_frame)
            
            if progress_callback:
                progress_callback(frame_idx + 1, total_frames)
    finally:
        cap.release()
        out.release()

def main():
    parser = argparse.ArgumentParser(description="Create a grayscale depth map video from a video file.")
    parser.add_argument("video_input", help="Path to the input video file.")
    parser.add_argument("video_output", help="Path to the output video file.")
    parser.add_argument("--model", default="depth-anything/DA3MONO-LARGE", help="The model to use for depth estimation.")
    parser.add_argument("--process-res", type=int, default=504, help="Processing resolution for the model.")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    model = DepthAnything3.from_pretrained(args.model).to(device).eval()

    try:
        cap = cv2.VideoCapture(args.video_input)
        if not cap.isOpened():
            raise IOError(f"Could not open video file {args.video_input}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        with tqdm(total=total_frames, desc=f"Processing {os.path.basename(args.video_input)}") as pbar:
            def progress_update(current, total):
                pbar.update(1)

            process_video(args.video_input, args.video_output, model, args.process_res, progress_callback=progress_update)
        
        print(f"Grayscale depth video saved to {args.video_output}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
