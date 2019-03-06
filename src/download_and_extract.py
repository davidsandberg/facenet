import requests
import zipfile
import os

model_dict = {
    'lfw-subset':      '1B5BQUZuJO-paxdN8UclxeHAR1WnR_Tzi', 
    '20170131-234652': '0B5MzpY9kBtDVSGM0RmVET2EwVEk',
    '20170216-091149': '0B5MzpY9kBtDVTGZjcWkzT3pldDA',
    '20170512-110547': '0B5MzpY9kBtDVZ2RpVDYwWmxoSUk',
    '20180402-114759': '1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-'
    }

def download_and_extract_file(model_name, data_dir):
    file_id = model_dict[model_name]
    destination = os.path.join(data_dir, model_name + '.zip')
    if not os.path.exists(destination):
        print('Downloading file to %s' % destination)
        download_file_from_google_drive(file_id, destination)
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            print('Extracting file to %s' % data_dir)
            zip_ref.extractall(data_dir)

def download_file_from_google_drive(file_id, destination):
    
        URL = "https://drive.google.com/uc?export=download"
    
        session = requests.Session()
    
        response = session.get(URL, params = { 'id' : file_id }, stream = True)
        token = get_confirm_token(response)
    
        if token:
            params = { 'id' : file_id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)
    
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
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
