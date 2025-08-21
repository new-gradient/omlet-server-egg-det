#!/usr/bin/env python3
"""
Training script for egg detection using RT-DETR v2
"""

import sys
import os
import argparse
from pathlib import Path

# Add RT-DETR to path
sys.path.insert(0, str(Path(__file__).parent / "RT-DETR" / "rtdetrv2_pytorch"))

def main():
    from tools.train import main as train_main
    
    parser = argparse.ArgumentParser(description='Train RT-DETR on egg detection dataset')
    parser.add_argument('--config', '-c', 
                        default='/home/ewan/omlet-server-egg-det/egg_detection_config.yml',
                        help='config file path')
    parser.add_argument('--output-dir', 
                        default='/home/ewan/omlet-server-egg-det/outputs/egg_detection_training',
                        help='output directory')
    parser.add_argument('--tuning', 
                        default='/home/ewan/omlet-server-egg-det/weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth',
                        help='tuning from checkpoint')
    parser.add_argument('--resume', type=str, 
                        help='resume from checkpoint')
    parser.add_argument('--test-only', action='store_true', 
                        help='test only')
    parser.add_argument('--use-amp', action='store_true', 
                        help='use automatic mixed precision')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--summary-dir', type=str,
                        help='tensorboard summary dir')
    parser.add_argument('--update', nargs='+', 
                        help='update yaml config')
    parser.add_argument('--print-method', type=str, default='builtin', 
                        help='print method')
    parser.add_argument('--print-rank', type=int, default=0, 
                        help='print rank id')
    parser.add_argument('--local-rank', type=int, 
                        help='local rank id')
    
    args = parser.parse_args()
    
    # Change to RT-DETR directory
    original_cwd = os.getcwd()
    os.chdir(Path(__file__).parent / "RT-DETR" / "rtdetrv2_pytorch")
    
    try:
        # Run training
        train_main(args)
    finally:
        # Change back to original directory
        os.chdir(original_cwd)

if __name__ == '__main__':
    main()