#!/bin/bash

set -e

if [ -z "$2" ]; then
  >&2 echo "Usage: ./find_matches.sh <MODEL_DIR> <DATA_DIR>"
  exit 1
fi

set -u

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

echo "> Building docker containers..."
(cd .. && docker build -t facenet .) >/dev/null
(cd find_matches && docker build -t facenet-find-matches .) >/dev/null

mkdir -p output
touch output/feature_vectors.json

echo "> Extracting feature vectors from $2 using $1..."
docker run --rm \
  -v $(realpath $1):/model:ro \
  -v $(realpath $2):/data:ro \
  -v $(PWD)/output/feature_vectors.json:/output/feature_vectors.json \
  facenet \
  python -m facenet.util.faces_to_vectors --inpath /data --outpath /output/feature_vectors.json --mdlpath /model

echo "> Starting MemSQL docker container..."
MEMSQL_CONTAINER_ID=$(cd find_matches && ./start_memsql.sh)
echo "> Waiting for memsql to start up..."

TIMER=0
until docker logs $MEMSQL_CONTAINER_ID | grep "Replaying logs/" > /dev/null
  do
    printf '.'
    sleep 1
    TIMER=$((TIMER+1))
    if [ $TIMER -gt 120 ]
      then
        echo
        echo "Timeout waiting for memsql to start up"
        exit 1
    fi
done

sleep 5

echo
echo "> Memsql successfully started"

echo "> Loading data into MemSQL..."
docker run --rm \
  --network=host \
  -v $(PWD)/find_matches:/find_matches:ro \
  -v $(PWD)/output/feature_vectors.json:/data/feature_vectors.json:ro \
  facenet-find-matches \
  bash -c "cd /find_matches && python load_data.py /data/feature_vectors.json"

echo "> Writing results to output/results.csv..."
docker run --rm \
  --network=host \
  -v $(PWD)/find_matches:/find_matches:ro \
  facenet-find-matches \
  bash -c "cd /find_matches && python find_matches.py 0.67" \
  > output/results.csv
