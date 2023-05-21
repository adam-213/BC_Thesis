import subprocess
import time

from a_preprocess_worker import Preprocessor
import time

if __name__ == '__main__':
    # time.sleep(7200)
    step = 400
    script_name = "a_preprocess_worker.py"


    source, target = "RawDS", "RevertDS"
    preprocessor = Preprocessor(source, target)
    preprocessor.load_scan_paths()

    for i in range(0, len(preprocessor.scans), step):
        subprocess.run(["python", script_name, "--start", str(i), "--number", str(step),
                        '--source', source, '--target', target], shell=True)

    print('Done')
