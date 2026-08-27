from src.predict import mode_predict
from src.train import train, mode_full_evaluation
import sys

def main():

    if len(sys.argv) == 1:
        mode_full_evaluation()

    elif len(sys.argv) == 4:
        subject = int(sys.argv[1])
        run = int(sys.argv[2])
        action = sys.argv[3]

        if action == "train":
            train(subject, run)
        elif action == "predict":
            mode_predict(subject, run)
        
        else:
            print("Unknown action. Use 'train' or 'predict'.")
            sys.exit(127)
 
    else:
        print("Usage:")
        print("  python3 mybci.py <subject> <run> train")
        print("  python3 mybci.py <subject> <run> predict")
        print("  python3 mybci.py")
        sys.exit(1)

if __name__ == "__main__":
    main()