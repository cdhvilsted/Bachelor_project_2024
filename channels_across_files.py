import numpy as np
import os
#from read_files import LoadFile
from get_annot_data import CountChannels

path ="EDF filer/"
folder = os.fsencode(path)
channels, channels_len, missing_chan = CountChannels(folder=folder)
possible_chan = set()
split_people = []

for i in range(len(channels_len)):
    if len(channels_len[i])==0 or len(channels_len[i])==1:
        print(i, ' has ', channels_len[i], ' channels')
        print(channels[i])
        print('-----------------')
        if len(channels_len[i])==1:
            possible_chan.update(channels[i][0])
    else:
        if len(channels_len[i]) != 1:
            if len(set(channels_len[i])) != 1:
                chan_set = set()
                for j in channels[i]:
                    chan_set.update(j)
                    possible_chan.update(j)
                
                split_people.append(i)
                print('person ', i, 'has uneven number of channels across splits')
                print(channels_len[i])                
                print(chan_set)
                print('-----------------')   
            else:
                chan_set = set()
                for j in channels[i]:
                    chan_set.update(j)
                    possible_chan.update(j)
                if len(chan_set) != channels_len[i][0]:
                    split_people.append(i)
                    print('person ', i, ' does not have the same channels across splits')
                    print(channels_len[i])
                    print(chan_set)
                    print('-----------------')
                else:
                    print('person ', i, 'has same channels across splits')
                    print(channels_len[i])
                    print(chan_set)
                    print('-----------------')

print(possible_chan)
print(split_people)
print('-----channels------')
print([channels[i] for i in split_people])