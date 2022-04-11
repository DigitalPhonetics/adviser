###############################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
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

import copy
import os
from typing import List, Type

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from services.policy.rl.policy_rl import RLPolicy
from services.policy.rl.dqn import DQN, DuelingDQN, NetArchitecture
from services.policy.rl.experience_buffer import Buffer, NaivePrioritizedBuffer
from services.service import Service, PublishSubscribe
from services.simulator.goal import Goal
from utils import common
from utils.beliefstate import BeliefState
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils.logger import DiasysLogger
from utils.sysact import SysAct, SysActionType
from utils.useract import UserActionType


class DQNPolicy(RLPolicy, Service):

    def __init__(self, domain: JSONLookupDomain,
                 architecture: NetArchitecture = NetArchitecture.DUELING,
                 hidden_layer_sizes: List[int] = [256, 700, 700],  # vanilla architecture
                 shared_layer_sizes: List[int] = [256], value_layer_sizes: List[int] = [300, 300],
                 advantage_layer_sizes: List[int] = [400, 400],  # dueling architecture
                 lr: float = 0.0001, discount_gamma: float = 0.99,
                 target_update_rate: int = 3,
                 replay_buffer_size: int = 8192, batch_size: int = 64,
                 buffer_cls: Type[Buffer] = NaivePrioritizedBuffer,
                 eps_start: float = 0.3, eps_end: float = 0.0,
                 l2_regularisation: float = 0.0, gradient_clipping: float = 5.0,
                 p_dropout: float = 0.0, training_frequency: int = 2, train_dialogs: int = 1000,
                 include_confreq: bool = False, logger: DiasysLogger = DiasysLogger(),
                 max_turns: int = 25,
                 summary_writer: SummaryWriter = None, device=torch.device('cpu')):
        """
        Args:
            target_update_rate: if 1, vanilla dqn update
                                if > 1, double dqn with specified target update
                                rate
        """
        RLPolicy.__init__(
            self,
            domain, buffer_cls=buffer_cls,
            buffer_size=replay_buffer_size, batch_size=batch_size,
            discount_gamma=discount_gamma, include_confreq=include_confreq,
            logger=logger, max_turns=max_turns, device=device)

        Service.__init__(self, domain=domain)

        self.writer = summary_writer
        self.training_frequency = training_frequency
        self.train_dialogs = train_dialogs
        self.lr = lr
        self.gradient_clipping = gradient_clipping
        if gradient_clipping > 0.0 and self.logger:
            self.logger.info("Gradient Clipping: " + str(gradient_clipping))
        self.target_update_rate = target_update_rate

        self.epsilon_start = eps_start
        self.epsilon_end = eps_end

        # Select network architecture
        if architecture == NetArchitecture.VANILLA:
            if self.logger:
                self.logger.info("Architecture: Vanilla")
            self.model = DQN(self.state_dim, self.action_dim,
                             hidden_layer_sizes=hidden_layer_sizes,
                             dropout_rate=p_dropout)
        else:
            if self.logger:
                self.logger.info("Architecture: Dueling")
            self.model = DuelingDQN(self.state_dim, self.action_dim,
                                    shared_layer_sizes=shared_layer_sizes,
                                    value_layer_sizes=value_layer_sizes,
                                    advantage_layer_sizes=advantage_layer_sizes,
                                    dropout_rate=p_dropout)
        # Select network update
        self.target_model = None
        if target_update_rate > 1:
            if self.logger:
                self.logger.info("Update: Double")
            if architecture == NetArchitecture.VANILLA:
                self.target_model = copy.deepcopy(self.model)
        elif self.logger:
            self.logger.info("Update: Vanilla")

        self.optim = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2_regularisation)
        self.loss_fun = nn.SmoothL1Loss(reduction='none')
        # self.loss_fun = nn.MSELoss(reduction='none')

        self.train_call_count = 0
        self.total_train_dialogs = 0
        self.epsilon = self.epsilon_start
        self.turns = 0
        self.cumulative_train_dialogs = -1

    def dialog_start(self, dialog_start=False):
        self.turns = 0
        self.last_sys_act = None
        if self.is_training:
            self.cumulative_train_dialogs += 1
        self.sys_state = {
            "lastInformedPrimKeyVal": None,
            "lastActionInformNone": False,
            "offerHappened": False,
            'informedPrimKeyValsSinceNone': []}

    def select_action_eps_greedy(self, state_vector: torch.FloatTensor):
        """ Epsilon-greedy policy.

        Args:
            state_vector (torch.FloatTensor): current state (dimension 1 x state_dim)

        Returns:
            action index for action selected by the agent for the current state
        """
        self.eps_scheduler()

        # epsilon greedy exploration
        if self.is_training and common.random.random() < self.epsilon:
            next_action_idx = common.random.randint(0, self.action_dim - 1)
        else:
            torch.autograd.set_grad_enabled(False)
            q_values = self.model(state_vector)
            next_action_idx = q_values.squeeze(dim=0).max(dim=0)[1].item()
            torch.autograd.set_grad_enabled(True)
        return next_action_idx

    @PublishSubscribe(sub_topics=["sim_goal"])
    def end(self, sim_goal: Goal):
        """
            Once the simulation ends, need to store the simulation goal for evaluation

            Args:
                sim_goal (Goal): the simulation goal, needed for evaluation
        """
        self.sim_goal = sim_goal

    def dialog_end(self):
        """
            clean up needed at the end of a dialog
        """
        self.end_dialog(self.sim_goal)
        if self.is_training:
            self.total_train_dialogs += 1
        self.train_batch()

    @PublishSubscribe(sub_topics=["beliefstate"], pub_topics=["sys_act", "sys_state"])
    def choose_sys_act(self, beliefstate: BeliefState = None) -> dict(sys_act=SysAct):
        """
            Determine the next system act based on the given beliefstate

            Args:
                beliefstate (BeliefState): beliefstate, contains all information the system knows
                                           about the environment (in this case the user)

            Returns:
                (dict): dictionary where the keys are "sys_act" representing the action chosen by
                        the policy, and "sys_state" which contains additional informatino which might
                        be needed by the NLU to disambiguate challenging utterances.
        """

        self.num_dialogs = self.cumulative_train_dialogs % self.train_dialogs
        if self.cumulative_train_dialogs == 0 and self.target_model is not None:
            # start with same weights for target and online net when a new epoch begins
            self.target_model.load_state_dict(self.model.state_dict())
        self.turns += 1
        if self.turns == 1:
            # first turn of dialog: say hello & don't record
            out_dict = self._expand_hello()
            out_dict["sys_state"] = {"last_act": out_dict["sys_act"]}
            return out_dict

        if self.turns > self.max_turns:
            # reached turn limit -> terminate dialog
            bye_action = SysAct()
            bye_action.type = SysActionType.Bye
            self.last_sys_act = bye_action
            # self.end_dialog(sim_goal)
            if self.logger:
                self.logger.dialog_turn("system action > " + str(bye_action))
            sys_state = {"last_act": bye_action}
            return {'sys_act': bye_action, "sys_state": sys_state}

        # intermediate or closing turn
        state_vector = self.beliefstate_dict_to_vector(beliefstate)
        next_action_idx = -1

        # check if user ended dialog
        if UserActionType.Bye in beliefstate["user_acts"]:
            # user terminated current dialog -> say bye
            next_action_idx = self.action_idx(SysActionType.Bye.value)
        if next_action_idx == -1:
            # dialog continues
            next_action_idx = self.select_action_eps_greedy(state_vector)

        self.turn_end(beliefstate, state_vector, next_action_idx)

        # Update the sys_state
        if self.last_sys_act.type in [SysActionType.InformByName, SysActionType.InformByAlternatives]:
            values = self.last_sys_act.get_values(self.domain.get_primary_key())
            if values:
                # belief_state['system']['lastInformedPrimKeyVal'] = values[0]
                self.sys_state['lastInformedPrimKeyVal'] = values[0]
        elif self.last_sys_act.type == SysActionType.Request:
            if len(list(self.last_sys_act.slot_values.keys())) > 0:
                self.sys_state['lastRequestSlot'] = list(self.last_sys_act.slot_values.keys())[0]

        self.sys_state["last_act"] = self.last_sys_act
        return {'sys_act': self.last_sys_act, "sys_state": self.sys_state}

    def _forward(self, state: torch.FloatTensor, action: torch.LongTensor):
        """ Forward state through DQN, return only Q-values for given actions.

        Args:
            state (torch.FloatTensor): states (dimension batch x state_dim)
            action (torch.LongTensor): actions to select Q-value for (dimension batch x 1)

        Returns:
            Q-values for selected actions
        """
        q_values = self.model(state)
        return q_values.gather(1, action)

    def _forward_target(self, state: torch.FloatTensor, reward: torch.FloatTensor,
                        terminal: torch.FloatTensor, gamma: float):
        """ Calculate target for TD-loss (DQN)

        Args:
            state (torch.FloatTensor): states (dimension batch x state_dim)
            reward (torch.FloatTensor): rewards (dimension batch x 1)
            terminal (torch.LongTensor): indicator {0,1} for terminal states (dimension: batch x 1)
            gamma (float): discount factor

        Returns:
            TD-loss targets
        """
        target_q_values = self.model(state)
        greedy_actions = target_q_values.max(1)[1].unsqueeze(1)
        return reward + (1.0 - terminal) * gamma * target_q_values.gather(1, greedy_actions)

    def _forward_target_ddqn(self, state: torch.FloatTensor, reward: torch.FloatTensor,
                             terminal: torch.FloatTensor, gamma: float):
        """ Calculate target for TD-loss (Double DQN - uses online and target network)

        Args:
            state (torch.FloatTensor): states (dimension batch x state_dim)
            reward (torch.FloatTensor): rewards (dimension batch x 1)
            terminal (torch.FloatTensor): indicator {0,1} for terminal states (dimension: batch x 1)
            gamma (float): discount factor

        Returns:
            TD-loss targets
        """
        greedy_actions = self.model(state).max(1)[1].unsqueeze(1)
        target_q_values = self.target_model(state).gather(1, greedy_actions)
        target_q_values = reward + (1.0 - terminal) * gamma * target_q_values
        return target_q_values

    def loss(self, s_batch: torch.FloatTensor, a_batch: torch.LongTensor,
             s2_batch: torch.FloatTensor, r_batch: torch.FloatTensor, t_batch: torch.FloatTensor,
             gamma: float):
        """ Calculate TD-loss for given experience tuples

        Args:
            s_batch (torch.FloatTensor): states (dimension batch x state_dim)
            a_batch (torch.LongTensor): actions (dimension batch x 1)
            s2_batch (torch.FloatTensor): next states (dimension: batch x state_dim)
            r_batch (torch.FloatTensor): rewards (dimension batch x 1)
            t_batch (torch.FloatTensor): indicator {0,1} for terminal states (dimension: batch x 1)
            gamma (float): discount factor

        Returns:
            TD-loss
        """
        # forward value
        torch.autograd.set_grad_enabled(True)
        q_val = self._forward(s_batch, a_batch)

        # forward target
        torch.autograd.set_grad_enabled(False)
        if self.target_model is None:
            q_target = self._forward_target(s2_batch, r_batch, t_batch, gamma)
        else:
            q_target = self._forward_target_ddqn(s2_batch, r_batch, t_batch,
                                                 gamma)
        torch.autograd.set_grad_enabled(True)

        # loss
        loss = self.loss_fun(q_val, q_target)
        return loss

    def train_batch(self):
        """ Train on a minibatch drawn from the experience buffer. """
        if not self.is_training:
            return

        if len(self.buffer) >= self.batch_size * 10 and \
                self.total_train_dialogs % self.training_frequency == 0:
            self.train_call_count += 1

            s_batch, a_batch, r_batch, s2_batch, t_batch, indices, importance_weights = \
                self.buffer.sample()

            self.optim.zero_grad()
            torch.autograd.set_grad_enabled(True)
            s_batch.requires_grad_()
            gamma = torch.tensor([self.discount_gamma] * self.batch_size, dtype=torch.float,
                                 device=self.device).view(self.batch_size, 1)

            # calculate loss
            loss = self.loss(s_batch, a_batch, s2_batch, r_batch, t_batch, gamma)
            if importance_weights is not None:
                loss = loss * importance_weights
                for i in range(self.batch_size):
                    # importance weighting
                    # update priorities
                    self.buffer.update(i, loss[i].item())
            loss = loss.mean()
            loss.backward()

            # clip gradients
            if self.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

            # update weights
            self.optim.step()
            current_loss = loss.item()
            torch.autograd.set_grad_enabled(False)

            if self.writer is not None:
                # plot loss
                self.writer.add_scalar('train/loss', current_loss, self.train_call_count)
                # plot min/max gradients
                max_grad_norm = -1.0
                min_grad_norm = 1000000.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        # TODO decide on norm
                        current_grad_norm = torch.norm(param.grad, 2)
                        if current_grad_norm > max_grad_norm:
                            max_grad_norm = current_grad_norm
                        if current_grad_norm < min_grad_norm:
                            min_grad_norm = current_grad_norm
                self.writer.add_scalar('train/min_grad', min_grad_norm, self.train_call_count)
                self.writer.add_scalar('train/max_grad', max_grad_norm, self.train_call_count)

            # update target net
            if self.target_model is not None and \
                    self.train_call_count % self.target_update_rate == 0:
                self.target_model.load_state_dict(self.model.state_dict())

    def eps_scheduler(self):
        """ Linear epsilon decay """
        if self.is_training:
            self.epsilon = max(0,
                               self.epsilon_start - (self.epsilon_start - self.epsilon_end)
                               * float(self.num_dialogs) / float(self.train_dialogs))
            if self.writer is not None:
                self.writer.add_scalar('train/eps', self.epsilon, self.total_train_dialogs)

    def save(self, path: str = os.path.join('models', 'dqn'), version: str = "1.0"):
        """ Save model weights

        Args:
            path (str): path to model folder
            version (str): appendix to filename, enables having multiple models for the same domain
                           (or saving a model after each training epoch)
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        model_file = os.path.join(
            path, "rlpolicy_" + self.domain.get_domain_name() + "_" + version + ".pt")
        torch.save(self.model, model_file)

    def load(self, path: str = os.path.join('models', 'dqn'), version: str = "1.0"):
        """ Load model weights

        Args:
            path (str): path to model folder
            version (str): appendix to filename, enables having multiple models for the same domain
                           (or saving a model after each training epoch)
        """
        model_file = os.path.join(
            path, "rlpolicy_" + self.domain.get_domain_name() + "_" + version + ".pt")
        if not os.path.isfile(model_file):
            raise FileNotFoundError("Could not find DQN policy weight file ", model_file)
        self.model = torch.load(model_file)
        self.logger.info("Loaded DQN weights from file " + model_file)
        if self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())

    def train(self, train=True):
        """ Sets module and its subgraph to training mode """
        super(DQNPolicy, self).train()
        self.is_training = True
        self.model.train()
        if self.target_model is not None:
            self.target_model.train()

    def eval(self, eval=True):
        """ Sets module and its subgraph to eval mode """
        super(DQNPolicy, self).eval()
        self.is_training = False
        self.model.eval()
        if self.target_model is not None:
            self.target_model.eval()
