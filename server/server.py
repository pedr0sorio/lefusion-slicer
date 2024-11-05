from flask import Flask, request, send_file
import shutil
from pathlib import Path
import subprocess
from datetime import datetime
import numpy as np
import os

app = Flask(__name__)

# Server Paths
DATA_DIR = Path('data')
IN_DATA_DIR = DATA_DIR / 'in'
OUT_DATA_DIR = DATA_DIR / 'out' # TODO ensure the out files are saved here

@app.route('/run_script', methods=['POST'])
def run_script():
    # Path to now uploaded .npz file with whole scan + bboxes idx
    img_path = request.form.get('input')
    # Lesion intensity/texture control
    histogram = request.form.get('histogram')
    # NOTE Add other hps from data_dict in inpainting helper
    # ...

    # Run tests
    npz_data = np.load( IN_DATA_DIR / img_path, 'r', allow_pickle=True)
    whole_scan = npz_data['imgs']
    bboxes = npz_data['boxes_numpy']

    print(f'Whole scan shape: {whole_scan.shape}')
    print(f'Whole scan max: {np.max(whole_scan)}')
    print(f'Whole scan min: {np.min(whole_scan)}')

    # # TODO Run inpainting and save to OUT_DATA_DIR
    # inpaintVolume(scan = whole_scan, bbox = bboxes, histogram = histogram, out_dir = OUT_DATA_DIR)

    return 'Script ran successfully'


@app.route('/download_file', methods=['GET'])
def download_file():
    output_name = request.form.get('output')
    # this is expected to be already the full path of the output file in the server
    return send_file(output_name, as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload_file():    
    file = request.files['file']

    if file:
        save_path = IN_DATA_DIR / file.filename
        os.makedirs(IN_DATA_DIR, exist_ok=True)
        file.save(save_path)
        return 'File uploaded successfully to %s' % save_path

@app.route('/upload_model', methods=['POST'])
def upload_model():    
    file = request.files['file']
    model_name = os.path.basename(file.filename).split('.')[0]
    checkpoint_dir = "./checkpoints/2.1/%s"%model_name

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    file.save(os.path.join(checkpoint_dir, os.path.basename(file.filename)))
    return 'Model uploaded successfully'


@app.route('/test_server', methods=['GET'])
def test():
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f'Server is up and running as of {date}'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
