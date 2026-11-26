from pathlib import Path

from ultralytics import YOLO


def main():
    project = Path(__file__).resolve().parent
    image_folder = project / "predict"
    weights = project / "runs" / "yolo12_training2" / "weights" / "best.pt"

    if not image_folder.is_dir():
        raise FileNotFoundError(f"Image folder missing: {image_folder}")
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {weights}")

    print(f"Using checkpoint: {weights}")
    model = YOLO(str(weights))

    results = model.predict(
        source=str(image_folder),
        device="cpu",
        imgsz=320,
        conf=0.05,
        save=True,
        stream=True,
        project=str(project / "runs"),
        name="predict_conf005",
        verbose=False,
    )

    for result in results:
        print(f"{Path(result.path).name}: {len(result.boxes)} detections")

    print(f"\nAnnotated images saved in: {model.predictor.save_dir}")


if __name__ == "__main__":
    main()