###############################################################################
#
# Copyright 2019, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################

import os
import sys
import argparse

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(get_root_dir())

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from modules.bst.ml.dstc_data import User, System, DialogTurn, Dialog, DSTC2Data
from modules.bst.ml.batch_provider import BatchProvider
from modules.bst.ml.belief_tracker import InformableTracker, RequestableTracker
import modules.bst.ml.config_helper as ch

# DOMAIN = JSONLookupDomain('CamRestaurants', '/resources/CamRestaurants-rules.json', '/resources/CamRestaurants-dbase.db')

# TODO create synthetic data by randomly replacing values!
# TODO batch
# TODO report joined accuracy

# train
def train_informables(writer, slot_name, epochs, batchsize, lr, l2, p_dropout, wordembdim, suffix, train=True, suppress_prints=False, dia_data=None, load_weights=False):
    if dia_data == None:
        dia_data = DSTC2Data()
    slot_value_count = dia_data.count_informable_slot_values(slot_name)
   
    if suppress_prints == False:
        print("training state tracker for slot ", slot_name)

    
    word_embedding_dim = wordembdim
    glove_embedding_dim = 300
    dense_hidden_dim = 300
    word_lvl_gru_hidden_dim = 100

    modelBelief = InformableTracker(dia_data.get_vocabulary(), slot_name, 
                                  slot_value_count,
                                  glove_embedding_dim=300, gru_dim=100, dense_output_dim=50,
                                  p_dropout=0.5)
    if len(suffix) > 0:
        modelBelief.weight_file_name = modelBelief.weight_file_name + "." + suffix
    if load_weights == True:
        modelBelief.load()
    modelBelief.to(ch.DEVICE)

    loss_fun = nn.CrossEntropyLoss()

    best_acc = -1.0
    for epoch in range(0, epochs):
        # train
        if suppress_prints == False:
            print("------------------------------")
            print("epoch " + str(epoch) + " ...")
            print("------------------------------")
        if train == True:
            modelBelief.train()
            train_epoch_loss, train_epoch_acc = informable_eval(dia_data, dia_data.get_train_data(), dia_data.get_train_len(), modelBelief, slot_value_count, slot_name, loss_fun, batchsize, lr, l2, train=True)
        if suppress_prints == False:
            print( "train loss: " + str(train_epoch_loss))
            print("train accuracy: " + str(train_epoch_acc))

        # eval
        modelBelief.eval()
        eval_epoch_loss, eval_epoch_acc = informable_eval(dia_data, dia_data.get_dev_data(), dia_data.get_dev_len(), modelBelief, slot_value_count, slot_name, loss_fun, batchsize, lr, l2, train=False)
        if suppress_prints == False:
            print("eval loss: " + str(eval_epoch_loss))
            print("eval accuracy: " + str(eval_epoch_acc))
        
        # plot
        if writer is not None:
            if train == True:
                writer.add_scalar('tracker/train/epoch_loss/', train_epoch_loss, epoch)
                writer.add_scalar('tracker/train/epoch_acc/', train_epoch_acc, epoch)
            writer.add_scalar('tracker/eval/epoch_loss/', eval_epoch_loss, epoch)
            writer.add_scalar('tracker/eval/epoch_acc/', eval_epoch_acc, epoch)
        
        # save best weights
        if train == True and eval_epoch_acc > best_acc:
            best_acc = eval_epoch_acc
            modelBelief.save()

    # test
    print("### Testing ###")
    modelBelief.eval()
    test_epoch_loss, test_epoch_acc = informable_eval(dia_data, dia_data.get_test_data(), dia_data.get_test_len(), modelBelief, slot_value_count, slot_name, loss_fun, batchsize, lr, l2, train=False)
    if suppress_prints == False:
        print("test loss: " + str(test_epoch_loss))
        print("test accuracy: " + str(test_epoch_acc))
    
    # plot
    if writer is not None:
        writer.add_scalar('tracker/test/epoch_loss/', test_epoch_loss, epoch)
        writer.add_scalar('tracker/test/epoch_acc/', test_epoch_acc, epoch)
    if suppress_prints == False:
        print("done")


# if train=False, the network will be evaluated instead of trained
# returns epoch loss, epoch acc
def informable_eval(data_wrapper, data, data_size, net, slot_value_count, slot_name, loss_fun, batchsize, lr, l2, train=True):
    epoch_loss = 0.0
    epoch_acc = 0.0
    turn_counter = 0
    batch_provider = BatchProvider(data, data_size, shuffle=True, batchSize=batchsize)

    if train == True:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    
    
    dia_idx = 0
    for dia_batch in batch_provider:
        batch_counter = 0
        batch_loss = 0.0
        for current_dialog in dia_batch:
            current_dialog_len = len(current_dialog)
            current_value = '**NONE**'
            
            # create turn batch
            outputs = []
            targets = []
            probs = []
            
            for turn_idx in range(0, current_dialog_len):
                turn_counter += 1
                turn = current_dialog.turns[turn_idx]

                if slot_name in turn.user.user_goal:
                    current_value = turn.user.user_goal[slot_name]
                sys_input = turn.system.input_txt
                usr_input = turn.user.utterance

                targets.append(torch.tensor([data_wrapper.get_informable_slot_value_index(slot_name, current_value)], dtype=torch.long, device=ch.DEVICE))
                
                # forward turn
                probabilities = net.forward(sys_input, usr_input, first_turn=(turn_idx==0))
                outputs.append(probabilities)
                probs.append(probabilities.clone().detach())
            
            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            loss = loss_fun(outputs, targets)
            batch_loss = batch_loss + loss

            # eval
            epoch_loss += float(loss.item())
            epoch_acc += torch.cat(probs, dim=0).argmax(1).eq(targets).sum().item()
            
            dia_idx += 1
            batch_counter += 1
            if dia_idx % 250 == 0:
                print("traing dialog " + str(dia_idx))
        
        if train == True:
            optimizer.zero_grad()
            batch_loss = batch_loss / float(len(dia_batch))
            batch_loss.backward()
            optimizer.step()
            
        

    epoch_acc = epoch_acc / float(turn_counter)
    return epoch_loss, epoch_acc


# train
def train_requestable(writer, req_name, epochs, batchsize, lr, l2, p_dropout, charembdim, suffix, train=True, suppress_prints=False, dia_data=None, load_weights=False):
    if dia_data == None:
        dia_data = DSTC2Data()

    if suppress_prints == False:
        print("training state tracker for requestable slots...")
        print(dia_data.get_requestable_slots())

    glove_embedding_dim = 300
    dense_hidden_dim = 300
    word_lvl_gru_hidden_dim = 100

    modelBelief = RequestableTracker(dia_data.get_vocabulary(), req_name,
                                  glove_embedding_dim=300, gru_dim=100, 
                                  dense_output_dim=50, p_dropout=0.5)
    if len(suffix) > 0:
        modelBelief.weight_file_name = modelBelief.weight_file_name + "." + suffix
    if load_weights == True:
        modelBelief.load()
    modelBelief.to(ch.DEVICE)
    
    loss_fun = nn.CrossEntropyLoss()

    best_acc = -1.0
    for epoch in range(0, epochs):
        # train
        if suppress_prints == False:
            print( "------------------------------")
            print("epoch " + str(epoch) + " ...")
            print("------------------------------")
        if train == True:
            modelBelief.train()
            train_epoch_loss, train_epoch_acc = requestable_eval(dia_data, dia_data.get_train_data(), dia_data.get_train_len(), modelBelief, req_name, loss_fun, batchsize, lr, l2, train=True)
        if suppress_prints == False:
            print("train loss: " + str(train_epoch_loss))
            print("train accuracy: " + str(train_epoch_acc))

        # eval
        modelBelief.eval()
        eval_epoch_loss, eval_epoch_acc = requestable_eval(dia_data, dia_data.get_dev_data(), dia_data.get_dev_len(), modelBelief, req_name, loss_fun, batchsize, lr, l2, train=False)
        if suppress_prints == False:    
            print("eval loss: " + str(eval_epoch_loss))
            print("eval accuracy: " + str(eval_epoch_acc))
        
        # plot
        if writer is not None:
            if train == True:
                writer.add_scalar('tracker/train/epoch_loss/', train_epoch_loss, epoch)
                writer.add_scalar('tracker/train/epoch_acc/', train_epoch_acc, epoch)
            writer.add_scalar('tracker/eval/epoch_loss/', eval_epoch_loss, epoch)
            writer.add_scalar('tracker/eval/epoch_acc/', eval_epoch_acc, epoch)
        
        # save best weights
        if train == True and eval_epoch_acc > best_acc:
            best_acc = eval_epoch_acc
            modelBelief.save()
    
    # test
    print("### Testing ###")
    modelBelief.eval()
    test_epoch_loss, test_epoch_acc = requestable_eval(dia_data, dia_data.get_test_data(), dia_data.get_test_len(), modelBelief, req_name, loss_fun, batchsize, lr, l2, train=False)
    if suppress_prints == False:    
        print("test loss: " + str(test_epoch_loss))
        print("test accuracy: " + str(test_epoch_acc))
    
    # plot
    if writer is not None:
        writer.add_scalar('tracker/test/epoch_loss/', test_epoch_loss, epoch)
        writer.add_scalar('tracker/test/epoch_acc/', test_epoch_acc, epoch)
    if suppress_prints == False:
        print("done")



# if train=False, the network will be evaluated instead of trained
# returns epoch loss, epoch acc
def requestable_eval(data_wrapper, data, data_size, net, req_name, loss_fun, batchsize, lr, l2, train=True):
    epoch_loss = 0.0
    epoch_acc = 0.0
    turn_counter = 0
    batch_provider = BatchProvider(data, data_size, shuffle=True, batchSize=batchsize)

    if train == True:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    
    dia_idx = 0
    for dia_batch in batch_provider:
        batch_counter = 0
        batch_loss = 0.0
        found = False
        for current_dialog in dia_batch:
            for turn in current_dialog.turns:
                if req_name in turn.user.requested_slots:
                    found = True
            if not found:
                continue

            current_dialog_len = len(current_dialog)
            current_value = False

            # create turn batch 
            outputs = []
            targets = []
            probs = []

            for turn_idx in range(0, current_dialog_len):
                turn_counter += 1 
                turn = current_dialog.turns[turn_idx]

                sys_input = turn.system.input_txt
                usr_input = turn.user.utterance

                target = torch.zeros(1, dtype=torch.long, device=ch.DEVICE) # binary slots
                if req_name in turn.user.requested_slots:
                    target[0] = int(True)
                targets.append(target)

                # forward turn
                probabilities = net.forward(sys_input, usr_input, first_turn=(turn_idx==0)) # 1 x 2
                outputs.append(probabilities)
                probs.append(probabilities.clone().detach())
       
            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            loss = loss_fun(outputs, targets)
            batch_loss = batch_loss + loss

            # eval
            epoch_loss += float(loss.item()) 
            epoch_acc += torch.cat(probs, dim=0).argmax(1).eq(targets).sum().item() / float(current_dialog_len)
         
            dia_idx += 1
            batch_counter += 1
            if dia_idx % 250 == 0:
                print("traing dialog " + str(dia_idx))

        if train == True and found == True:
            optimizer.zero_grad()
            batch_loss = batch_loss / float(len(dia_batch))
            batch_loss.backward()
            optimizer.step()

    epoch_acc = epoch_acc / float(data_size)
    return epoch_loss, epoch_acc



# train
def train_method(writer, epochs, batchsize, lr, l2, p_dropout, wordembdim, suffix, train=True, suppress_prints=False, dia_data=None, load_weights=False):
    if dia_data == None:
        dia_data = DSTC2Data()
    slot_value_count = len(dia_data.get_method_slots())
   
    if suppress_prints == False:
        print("training state tracker for methods ")

    
    word_embedding_dim = wordembdim
    glove_embedding_dim = 300
    dense_hidden_dim = 300
    word_lvl_gru_hidden_dim = 100

    modelBelief = InformableTracker(dia_data.get_vocabulary(), 'methods', 
                                  slot_value_count,
                                  glove_embedding_dim=300, gru_dim=100, dense_output_dim=0,
                                  p_dropout=0.5)
    if len(suffix) > 0:
        modelBelief.weight_file_name = modelBelief.weight_file_name + "." + suffix
    if load_weights == True:
        modelBelief.load()
    modelBelief.to(ch.DEVICE)

    loss_fun = nn.CrossEntropyLoss()

    best_acc = -1.0
    for epoch in range(0, epochs):
        # train
        if suppress_prints == False:
            print("------------------------------")
            print("epoch " + str(epoch) + " ...")
            print("------------------------------")
        if train == True:
            modelBelief.train()
            train_epoch_loss, train_epoch_acc = method_eval(dia_data, dia_data.get_train_data(), dia_data.get_train_len(), modelBelief, slot_value_count, 'method', loss_fun, batchsize, lr, l2, train=True)
        if suppress_prints == False:
            print( "train loss: " + str(train_epoch_loss))
            print("train accuracy: " + str(train_epoch_acc))

        # eval
        modelBelief.eval()
        eval_epoch_loss, eval_epoch_acc = method_eval(dia_data, dia_data.get_dev_data(), dia_data.get_dev_len(), modelBelief, slot_value_count, 'method', loss_fun, batchsize, lr, l2, train=False)
        if suppress_prints == False:
            print("eval loss: " + str(eval_epoch_loss))
            print("eval accuracy: " + str(eval_epoch_acc))
        
        # plot
        if writer is not None:
            if train == True:
                writer.add_scalar('tracker/train/epoch_loss/', train_epoch_loss, epoch)
                writer.add_scalar('tracker/train/epoch_acc/', train_epoch_acc, epoch)
            writer.add_scalar('tracker/eval/epoch_loss/', eval_epoch_loss, epoch)
            writer.add_scalar('tracker/eval/epoch_acc/', eval_epoch_acc, epoch)
        
        # save best weights
        if train == True and eval_epoch_acc > best_acc:
            best_acc = eval_epoch_acc
            modelBelief.save()

    # test
    modelBelief.eval()
    test_epoch_loss, test_epoch_acc = method_eval(dia_data, dia_data.get_test_data(), dia_data.get_test_len(), modelBelief, slot_value_count, 'method', loss_fun, batchsize, lr, l2, train=False)
    if suppress_prints == False:
        print("test loss: " + str(test_epoch_loss))
        print("test accuracy: " + str(test_epoch_acc))
    
    # plot
    print("### Testing ###")
    if writer is not None:
        writer.add_scalar('tracker/test/epoch_loss/', test_epoch_loss, epoch)
        writer.add_scalar('tracker/test/epoch_acc/', test_epoch_acc, epoch)
    if suppress_prints == False:
        print("done")


# if train=False, the network will be evaluated instead of trained
# returns epoch loss, epoch acc
def method_eval(data_wrapper, data, data_size, net, slot_value_count, slot_name, loss_fun, batchsize, lr, l2, train=True):
    epoch_loss = 0.0
    epoch_acc = 0.0
    turn_counter = 0
    batch_provider = BatchProvider(data, data_size, shuffle=True, batchSize=batchsize)

    if train == True:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    
    
    dia_idx = 0
    for dia_batch in batch_provider:
        batch_counter = 0
        batch_loss = 0.0
        for current_dialog in dia_batch:
            current_dialog_len = len(current_dialog)
            
            # create turn batch
            outputs = []
            targets = []
            probs = []
            
            for turn_idx in range(0, current_dialog_len):
                turn_counter += 1
                turn = current_dialog.turns[turn_idx]
                method_name = turn.user.method
                sys_input = turn.system.input_txt
                usr_input = turn.user.delexicalised_utterance

                targets.append(torch.tensor([data_wrapper.get_method_index(method_name)], dtype=torch.long, device=ch.DEVICE))
                
                # forward turn
                probabilities = net.forward(sys_input, usr_input, first_turn=(turn_idx==0))
                outputs.append(probabilities)
                probs.append(probabilities.clone().detach())
            
            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            loss = loss_fun(outputs, targets)
            batch_loss = batch_loss + loss

            # eval
            epoch_loss += float(loss.item())
            epoch_acc += torch.cat(probs, dim=0).argmax(1).eq(targets).sum().item()
            
            dia_idx += 1
            batch_counter += 1
            if dia_idx % 250 == 0:
                print("traing dialog " + str(dia_idx))
        
        if train == True:
            optimizer.zero_grad()
            batch_loss = batch_loss / float(len(dia_batch))
            batch_loss.backward()
            optimizer.step()
            
        

    epoch_acc = epoch_acc / float(turn_counter)
    return epoch_loss, epoch_acc




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-infslot', default='food', help='name of informable slot')
    parser.add_argument('-epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('-batchsize', default=1, type=int, help='batch size ')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-l2', default=0.0001, type=float, help='l2 penalty')
    parser.add_argument('-charembdim', default=100, type=int, help='char embedding dimension >= 0 and divisible by 2')
    parser.add_argument('-gpu', default=0, type=int, help='gpu id')
    parser.add_argument('-cuda', default=False, action='store_true',  help='use cuda')
    parser.add_argument('-suffix', default='', help='appended to all output files - allows multiple training instances at once')
    #parser.add_argument('-test', default=False, action='store_true', help='quick test with custom test sentence')
    parser.add_argument('-reqslot', default=None, help='name of requestables slot instead of informable')
    parser.add_argument('-methods', default=False, action='store_true', help='train methods instead of default informables')
    parser.add_argument('-nowriter', default=False, action='store_true', help='if option is set, no tensorboard writer will be created')
    parser.add_argument('-dropout', default=0.5, type=float, help='dropout probability')
    opt = parser.parse_args()

    ch.set_device(opt.cuda, opt.gpu)
    print("device: ")
    print(ch.DEVICE)
    print("-------")
    writer = None

    if opt.reqslot is not None:
        if opt.nowriter == False:
            writer = SummaryWriter(log_dir=os.path.join(ch.get_runs_path(), 'requestable' + opt.suffix + 'slot_' + opt.reqslot))
        train_requestable(writer, opt.reqslot, opt.epochs, opt.batchsize, opt.lr, opt.l2, opt.dropout, opt.charembdim, opt.suffix)
        # train requestables
    elif opt.methods:
        if opt.nowriter == False:
            writer = SummaryWriter(log_dir=os.path.join(ch.get_runs_path(), 'method' + opt.suffix))
        train_method(writer, opt.epochs, opt.batchsize, opt.lr, opt.l2, opt.dropout, opt.charembdim, opt.suffix)
    elif opt.infslot is not None:
        if opt.nowriter == False:
            writer = SummaryWriter(log_dir=os.path.join(ch.get_runs_path(), 'informables' + opt.suffix, 'slot_' + opt.infslot))
        #if opt.test == True:
        #    quicktest(opt.cuda)
        #else:
        train_informables(writer, opt.infslot, opt.epochs, opt.batchsize, opt.lr, opt.l2, opt.dropout, opt.charembdim, opt.suffix)
