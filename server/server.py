from flask import Flask, request, send_file
import shutil
from pathlib import Path
import subprocess
from datetime import datetime
import numpy as np
import os
import select

from lefusion.paths import IN_SERVER_DATA_DIR, OUT_SERVER_DATA_DIR
os.makedirs(IN_SERVER_DATA_DIR, exist_ok=True)
os.makedirs(OUT_SERVER_DATA_DIR, exist_ok=True)

app = Flask(__name__)


@app.route('/run_script', methods=['POST'])
def run_script():
    # Path to now uploaded .npz file with whole scan + bboxes idx
    img_path_list = request.form.getlist('input')
    debug = request.form.get('debug')
    jump_length = request.form.get('jump_length')
    if jump_length is None:
        jump_length = 1
    jump_n_sample = request.form.get('jump_n_sample')
    if jump_n_sample is None:
        jump_n_sample = 1
    batch_size = request.form.get('batch_size')
    if batch_size is None:
        batch_size = 1

    inference_script_path = os.path.abspath('LeFusion_LIDC/test/inference.py')
    args = f"""
    python
    {inference_script_path}
    dataset_root_dir={IN_SERVER_DATA_DIR.as_posix()}
    target_img_path={OUT_SERVER_DATA_DIR.as_posix()}
    schedule_jump_params.jump_length={jump_length}
    schedule_jump_params.jump_n_sample={jump_n_sample}
    batch_size={batch_size}
    slicer=True
    """.replace("    ", "").split("\n")
    args = [arg for arg in args if arg != ""]
    if debug:
        print(f" ----------- {debug = } -----------")
        args.append("debug=True")


    process = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True
    )

    # Read stdout and stderr continuously
    while True:
        reads = [process.stdout.fileno(), process.stderr.fileno()]
        ret = select.select(reads, [], [])
        for fd in ret[0]:
            if fd == process.stdout.fileno():
                output = process.stdout.readline()
                if output:
                    print(output, end='', flush=True)
            if fd == process.stderr.fileno():
                error = process.stderr.readline()
                if error:
                    print(error, end='', flush=True)

        if process.poll() is not None:
            break

    # Ensure all remaining output is printed
    for output in process.stdout:
        print(output, end='', flush=True)
    for error in process.stderr:
        print(error, end='', flush=True)

    stderr = process.stderr.read()
    print('===========================\n', stderr, '\n===========================')
    
    if process.returncode == 0:
        return f'Success: {output.strip()}'
    else:
        return f'Error: {stderr.strip()}'


@app.route('/download_file', methods=['GET'])
def download_file():
    output_name = request.form.get('output')
    # this is expected to be already the full path of the output file in the server
    return send_file(output_name, as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload_file():    
    file = request.files['file']

    if file:
        save_path = IN_SERVER_DATA_DIR / file.filename
        os.makedirs(IN_SERVER_DATA_DIR, exist_ok=True)
        file.save(save_path)
        return 'File uploaded successfully to %s' % save_path

@app.route('/flush_memory', methods=['GET'])
def flush_memory():
    # Remove all .npz files from IN and OUT directories
    for directory_path in [IN_SERVER_DATA_DIR, OUT_SERVER_DATA_DIR]:
        for item in directory_path.iterdir():
            if item.name.endswith('.npz'):
                print(f'Removed .npz file: \n  -> {item}')
                item.unlink()

    return 'Server .npz files flushed. Both IN and OUT'


@app.route('/test_server', methods=['GET'])
def test():
    print(f'IN_SERVER_DATA_DIR: {IN_SERVER_DATA_DIR}')
    print(f'OUT_SERVER_DATA_DIR: {OUT_SERVER_DATA_DIR}')
    print(f'os.getcwd(): {os.getcwd()}')
    # Absolute path to the inference script
    inference_script_path = os.path.abspath('LeFusion_LIDC/test/inference.py')
    print(f'inference_script_path: {inference_script_path}')
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f'Server is up and running as of {date}'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
