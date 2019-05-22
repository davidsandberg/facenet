# Instructions

## With docker-compose

* Replace values in .env
* Run ```docker-compose up```

## Without docker

* The validation script requires python 3.6
    * If using [anaconda](https://www.anaconda.com/download/), create a virtual environment with ```conda create -n py36 python=3.6```
    * Activate the virtual environment with ```conda activate py36```
    * Install requirements with ```pip install -r requirements.txt```
* Example to run validation script:

```bash
python validate.py --image_dir /images --model_path /facenet.pb --distance_metric ANGULAR_DISTANCE --pairs_fname /pairs/pairs.txt --threshold_start 0 --threshold_end 4 --threshold_step 0.01 --embedding_size 128 --threshold_metric ACCURACY --prealigned_flag --remove_empty_embeddings_flag
--is_insightface
```
