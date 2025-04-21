import argparse
import time

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image

import networks
from layers import disp_to_depth

def load_model(weights_folder: str, model_name: str, device):
    # load checkpoints
    encoder_path = f"{weights_folder}/encoder.pth"
    decoder_path = f"{weights_folder}/depth.pth"
    encoder_ckpt = torch.load(encoder_path, map_location=device)
    decoder_ckpt = torch.load(decoder_path, map_location=device)

    # get training resolution
    feed_height = encoder_ckpt['height']
    feed_width = encoder_ckpt['width']

    # build and load encoder
    encoder = networks.LiteMono(model=model_name,
                                height=feed_height,
                                width=feed_width)
    enc_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_ckpt.items() if k in enc_dict})
    encoder.to(device).eval()

    # build and load decoder
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    dec_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_ckpt.items() if k in dec_dict})
    depth_decoder.to(device).eval()

    return encoder, depth_decoder, feed_width, feed_height

def main():
    parser = argparse.ArgumentParser(description="Live Depth from Mono Model")
    parser.add_argument("--weights_folder", type=str, required=True,
                        help="path to folder containing encoder.pth & depth.pth")
    parser.add_argument("--model_name", type=str, default="lite-mono",
                        choices=["lite-mono","lite-mono-small","lite-mono-tiny","lite-mono-8m"])
    parser.add_argument("--camera_id", type=int, default=0,
                        help="webcam device ID")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="target processing frames per second")
    args = parser.parse_args()

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load model
    encoder, depth_decoder, feed_w, feed_h = load_model(
        args.weights_folder, args.model_name, device)

    # set up transforms
    to_tensor = transforms.ToTensor()

    # open camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera_id}")

    delay = 1.0 / args.fps
    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # preprocess
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (feed_w, feed_h), interpolation=cv2.INTER_LANCZOS4)
            input_tensor = to_tensor(small).unsqueeze(0).to(device)

            # inference
            with torch.no_grad():
                features = encoder(input_tensor)
                outputs = depth_decoder(features)
                disp = outputs[("disp", 0)]

            # upsample disparity to original frame size
            disp_resized = F.interpolate(
                disp, (frame.shape[0], frame.shape[1]),
                mode="bilinear", align_corners=False
            )
            disp_np = disp_resized.squeeze().cpu().numpy()

            # normalize for visualization
            # vmax = np.percentile(disp_np, 95)
            # vmin = disp_np.min()
            # norm = np.clip((disp_np - vmin) / (vmax - vmin), 0, 1)
            # heat = (norm * 255).astype(np.uint8)
            # heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_MAGMA)

            # # overlay
            # overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

            # after you have disp_np (H×W numpy array of disparity values)
            block_size = 32  # 8×8 = 64 pixels

            # make a copy of the original frame to draw on:
            overlay = frame.copy()

            h, w = disp_np.shape
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    block = disp_np[y : min(y+block_size, h),
                                    x : min(x+block_size, w)]
                    if block.size == 0:
                        continue
                    mean_disp = float(block.mean())
                    # draw the number in white, centered in the block
                    cx = x + block_size // 2
                    cy = y + block_size // 2
                    cv2.putText(
                        overlay,
                        f"{mean_disp:.2f}",
                        (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,               # font scale
                        (255, 255, 255),   # white text
                        1,
                        cv2.LINE_AA,
                    )

            # display
            cv2.imshow("Live Depth Heatmap", overlay)

            # maintain target FPS
            elapsed = time.time() - start
            wait = max(1, int((delay - elapsed) * 1000))
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()