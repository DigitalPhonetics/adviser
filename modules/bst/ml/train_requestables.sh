# possible request slots: 'addr', 'area', 'food', 'name', 'phone', 'postcode', 'pricerange', 'signature'
nohup taskset 0xFF python -u train_trackers.py -reqslot addr -epochs 25 -lr 0.001 -dropout 0.5 -batchsize 1 -l2 0.001 -cuda -gpu 0 > log_req_addr.txt &
nohup taskset 0xFFFF python -u train_trackers.py -reqslot area -epochs 25 -lr 0.001 -dropout 0.5 -batchsize 1 -l2 0.001 -cuda -gpu 0 > log_req_area.txt &
nohup taskset 0xFFFF python -u train_trackers.py -reqslot food -epochs 25 -lr 0.001 -dropout 0.5 -batchsize 1 -l2 0.001 -cuda -gpu 0 > log_req_food.txt &
nohup taskset 0xFFFF python -u train_trackers.py -reqslot name -epochs 25 -lr 0.001 -dropout 0.5 -batchsize 1 -l2 0.001 -cuda -gpu 0 > log_req_name.txt &
nohup taskset 0xFFFF python -u train_trackers.py -reqslot phone -epochs 25 -lr 0.001 -dropout 0.5 -batchsize 1 -l2 0.001 -cuda -gpu 0 > log_req_phone.txt &
nohup taskset 0xFFFF python -u train_trackers.py -reqslot postcode -epochs 25 -lr 0.001 -dropout 0.5 -batchsize 1 -l2 0.001 -cuda -gpu 0 > log_req_postcode.txt &
nohup taskset 0xFFFF python -u train_trackers.py -reqslot pricerange -epochs 25 -lr 0.001 -dropout 0.5 -batchsize 1 -l2 0.001 -cuda -gpu 0 > log_req_pricerange.txt &
nohup taskset 0xFFFF python -u train_trackers.py -reqslot signature -epochs 25 -lr 0.001 -dropout 0.5 -batchsize 1 -l2 0.001 -cuda -gpu 0 > log_req_signature.txt &