#! /usr/bin/env python3
"""
Simple script to predict which type of flower an input image is of given a trained model.
"""
import argparse
import logging

from utils import predict_from_saved_model

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file_path", type=str,
                        help="File path for the jpeg image to classify")
    parser.add_argument("model_file_path", type=str,
                        help="File path for the saved model")
    parser.add_argument("--top_k", type=int, default=1, required=False,
                        help="Display the top <top_k> predicted classes")
    parser.add_argument("--category_names", type=str, default=None, required=False,
                        help="JSON input file specifying the category label to class label (e.g. '0': 'pink primrose')")
    args = parser.parse_args()

    return args


def main():
    """
    Main script
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # get command line arguments and run prediction
    args = parse_args()
    predict_from_saved_model(args)

    return


if __name__ == "__main__":
    main()