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

import random
import math


# Iterator for shuffling and batching data
class BatchProvider(object):
    def __init__(self, supervised_data, data_size, shuffle=True, batchSize=1):
        assert(batchSize > 0)
        self.supervised_data = supervised_data   
        self.batchSize = batchSize
        self.length = 0
        self.length = data_size
        self.indices = list(range(0, self.length))
        if shuffle == True:
            random.shuffle(self.indices)

    def __len__(self):
        return self.length
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.indices) == 0:
            raise StopIteration
        else:
            if self.batchSize == 1:
                return [self.supervised_data[self.indices.pop()]]
            else:
                batch = []
                for i in range(0, min(self.batchSize, len(self.indices))):
                    batch.append(self.supervised_data[self.indices.pop()])
                return batch


    
    
    
