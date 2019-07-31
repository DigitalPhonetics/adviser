#!/bin/bash
# get cam restaurant data
(cd ../../../resources/databases/ && curl -O https://bitbucket.org/dialoguesystems/pydial/raw/e586e871311a3f2d1b81b451835f620c303e4c4b/ontology/ontologies/CamRestaurants-rules.json)
(cd ../../../resources/databases/ && curl -O https://bitbucket.org/dialoguesystems/pydial/raw/e586e871311a3f2d1b81b451835f620c303e4c4b/ontology/ontologies/CamRestaurants-dbase.db)
# get dstc 2 data
echo "Downloading DSTC2 training data"
curl -O http://camdial.org/~mh521/dstc/downloads/dstc2_traindev.tar.gz
echo "Downloading DSTC2 test data"
curl -O http://camdial.org/~mh521/dstc/downloads/dstc2_test.tar.gz
# get glove embedding
echo "Downloading glove6B embeddings"
curl -L -O http://nlp.stanford.edu/data/glove.6B.zip
# create directories
mkdir data
mkdir data/dstc2
mkdir data/dstc2/traindev
mkdir data/dstc2/test
mkdir data/preprocessing
mkdir data/glove.6B
mkdir weights
# unpack
echo "Unpacking..."
tar -zxf dstc2_traindev.tar.gz --directory data/dstc2/traindev
tar -zxf dstc2_test.tar.gz --directory  data/dstc2/test
unzip glove.6B -d data/glove.6B
# cleanup
echo "Cleanup..."
rm dstc2_traindev.tar.gz
rm dstc2_test.tar.gz
rm glove.6B.zip
# download tokenizer language model
echo "Downloading spacy en language model"
python -m spacy download en_core_web_sm
# preprocessing
echo "Preprocessing..."
python -u DSTCData.py 
echo "########## DONE #############"

