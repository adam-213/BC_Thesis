import subprocess
import time

from a_preprocess_worker import Preprocessor
import time
if __name__ == '__main__':
    # time.sleep(7200)
    preprocessor = Preprocessor()
    preprocessor.load_scan_paths()
    step = 400

    script_name = "a_preprocess_worker.py"  # Replace with the name of your scrip
    for i in range(0, len(preprocessor.scans), step):
        subprocess.run(["python", script_name, "--start", str(i), "--number", str(step)], shell=True)

    print('Done')
