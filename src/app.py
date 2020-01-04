import os
import sys
import subprocess
import requests
import ssl
import random
import string
import json

from flask import jsonify
from flask import Flask
from flask import request
from flask import send_file
import traceback

from app_utils import blur
from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import get_multi_model_bin

import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms


try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus


app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/detect", methods=["POST"])
def detect():

    input_path = generate_random_filename(upload_directory,"jpg")

    try:
        
        if 'file' in request.files:
            file = request.files['file']
            if allowed_file(file.filename):
                file.save(input_path)
            try:
                top_k = request.form.getlist('top_k')[0]
            except:
                top_k = 5
            
        else:
            url = request.json["url"]
            download(url, input_path)

            try:
                top_k = request.json["top_k"]
            except:
                top_k = 5
       
        results = []
        
        img = tfms(Image.open(input_path)).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            outputs = model(img)

        for idx in torch.topk(outputs, k=int(top_k)).indices.squeeze(0).tolist():
            prob = torch.softmax(outputs, dim=1)[0, idx].item()
            labels = [x.strip() for x in labels_map[idx].split(',')]
            results.append({
                'label': labels[0],
                'labels': labels,
                'score': '{p:.2f}%'.format(p=prob*100)
                })

        return json.dumps(results), 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        clean_all([
            input_path
            ])

if __name__ == '__main__':
    global upload_directory, model_directory
    global model, labels_map
    global tfms
    global ALLOWED_EXTENSIONS
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

    upload_directory = '/src/upload/'
    create_directory(upload_directory)

    model_directory = '/src/model/'
    create_directory(model_directory)

    model_name = 'efficientnet-b5'
    model = EfficientNet.from_pretrained(model_name)
    model.eval()


    model_url = "https://storage.gra5.cloud.ovh.net/v1/AUTH_18b62333a540498882ff446ab602528b/pretrained-models/image/EfficientNet-PyTorch/"

    labels_file = 'labels_map.txt'

    get_model_bin(model_url + labels_file, model_directory + labels_file)

    labels_map = json.load(open(model_directory + labels_file))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=True)
    
