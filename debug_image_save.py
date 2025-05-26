import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Debug OpenCV image saving.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output", type=str, help="Path to save the output image. Defaults to 'debug_output.png' in the same directory as the input image.")
    args = parser.parse_args()

    # Read the image
    image = cv2.imread(args.input)

    if image is None:
        print(f"Error: Could not read image from {args.input}")
        return

    # Determine output path
    if args.output:
        output_path = args.output
        
        # Check if output_path is a directory
        if os.path.isdir(output_path):
            # If it's a directory, add a default filename
            input_filename = os.path.basename(args.input)
            name, _ = os.path.splitext(input_filename)
            output_path = os.path.join(output_path, f"{name}_debug.png")
        elif not os.path.splitext(output_path)[1]:
            # If no extension, add .png
            output_path += ".png"
    else:
        input_dir = os.path.dirname(args.input)
        if not input_dir: # Handle case where input is just filename
            input_dir = "."
        output_path = os.path.join(input_dir, "debug_output.png")
        
    # Ensure output directory exists
    output_dir_name = os.path.dirname(output_path)
    if output_dir_name and not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name, exist_ok=True)
        print(f"Created output directory: {output_dir_name}")

    print(f"Input image: {args.input}")
    print(f"Output path: {output_path}")

    # Save the image
    try:
        success = cv2.imwrite(output_path, image)
        if success:
            print(f"Successfully saved image to {output_path}")
        else:
            print(f"Error: Failed to save image to {output_path}")
    except Exception as e:
        print(f"An error occurred during cv2.imwrite: {e}")

if __name__ == "__main__":
    main() 