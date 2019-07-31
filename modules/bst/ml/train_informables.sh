# possible inform slots: 'name', 'area', 'pricerange', 'food'
nohup taskset 0xFF python -u train_trackers.py -slot food -epochs 100 -lr 0.001 -dropout 0.5 -batchsize 1 -l2 0.001 -cuda -gpu 0 > log_food.txt &
nohup taskset 0xFF python -u train_trackers.py -slot area -epochs 50 -lr 0.001 -dropout 0.5 -batchsize 1 -l2 0.001 -cuda -gpu 0 > log_area.txt &
nohup taskset 0xFF python -u train_trackers.py -slot pricerange -epochs 100 -lr 0.001 -dropout 0.5 -batchsize 1 -l2 0.001 -cuda -gpu 0 > log_price.txt &