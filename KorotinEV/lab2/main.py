import argparse
from app import VehicleDetectionApp


def parser():
    parser = argparse.ArgumentParser(description="Vehicle Detection using OpenCV DNN")
    parser.add_argument("--images", type=str, required=True, 
                       help="Path to images directory")
    parser.add_argument("--annotations", type=str, required=True,
                       help="Path to annotations file")
    parser.add_argument("--model", type=str, required=True,
                       choices=["yolo", "mobilenet", "rcnn"],
                       help="Model type for detection")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold for detection")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable image display")
    parser.add_argument("--show-ground-truth", action="store_true",
                       help="Display ground truth bounding boxes")

    return parser.parse_args()

def main():
    args = parser()
    
    app = VehicleDetectionApp()
    app.run(
        images_path=args.images,
        annotation_path=args.annotations,
        model_type=args.model,
        confidence_threshold=args.confidence,
        display=not args.no_display,
        show_ground_truth=args.show_ground_truth
    )

if __name__ == "__main__":
    main()
