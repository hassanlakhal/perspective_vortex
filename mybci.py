from src.predict import mode_predict
from src.train import train, mode_full_evaluation
import sys

def main():

    source = 'physionet'
    if len(sys.argv) in (1, 2):
        if len(sys.argv) == 2:
            source =  sys.argv[1]
        mode_full_evaluation(source)
    
    elif len(sys.argv) in (4, 5):

        subject = int(sys.argv[1])
        run = int(sys.argv[2])
        action = sys.argv[3]

        if len(sys.argv) == 5:
            source =  sys.argv[4]
        
        if action == "train":
            train(subject, run, source)
        elif action == "predict":
            mode_predict(subject, run, source)
        
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