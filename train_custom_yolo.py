from ultralytics import YOLO
import os
import shutil

def main():
    print("Loading base model...")
    model = YOLO('yolov8n.pt')

    # Ensure this points to your exact data.yaml file
    yaml_path = r"D:\Judo_Pipeline\AI Training Data\March 12\data.yaml"
    
    # Define directories inside your current project folder
    project_dir = os.getcwd()
    training_runs_dir = os.path.join(project_dir, "training_runs")

    print("Starting training...")
    results = model.train(
        data=yaml_path, 
        epochs=50, 
        imgsz=640, 
        device='cpu',  # Change to 'cpu' if no GPU is available
        project=training_runs_dir,
        name="judo_model"
    )
    
    # Locate the generated weights and copy them to the root project folder
    best_pt_source = os.path.join(training_runs_dir, "judo_model", "weights", "best.pt")
    final_model_dest = os.path.join(project_dir, "judo_custom_v1.pt")
    
    if os.path.exists(best_pt_source):
        shutil.copy(best_pt_source, final_model_dest)
        print(f"\nSUCCESS! Custom model exported to: {final_model_dest}")
    else:
        print("\nTraining finished, but best.pt was not found. Check the training_runs directory.")

if __name__ == '__main__':
    main()