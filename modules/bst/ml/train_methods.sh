nohup taskset 0xF python -u train_trackers.py -methods -epochs 25 -lr 0.001 -dropout 0.5 -batchsize 1 -l2 0.001 -cuda -gpu 0 > log_methods.txt 
