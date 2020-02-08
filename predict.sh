rm /cinnamon_ai_marathon/tmp/*

./preprocess.sh /cinnamon_ai_marathon/input/ /cinnamon_ai_marathon/tmp/
python3 predict.py --model /cinnamon_ai_marathon/model/ --data /cinnamon_ai_marathon/tmp/ --output /cinnamon_ai_marathon/output/predict.json --device -1

rm /cinnamon_ai_marathon/tmp/*
