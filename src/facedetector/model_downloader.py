import requests
import os
from os.path import expanduser
import zipfile


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download():
    file_id = '1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-'
    dl_path = './.facenet_model/'
    if not os.path.exists(dl_path):
        os.makedirs(dl_path)
    destination = dl_path + 'model.zip'
    download_file_from_google_drive(file_id, destination)

    zip_ref = zipfile.ZipFile(destination, 'r')
    zip_ref.extractall(dl_path)
    zip_ref.close()
    os.remove('./.facenet_model/model.zip')