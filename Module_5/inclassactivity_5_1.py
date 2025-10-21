# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 09:00:55 2025

@author: jcmir
"""

import numpy as np

three_d_array = np.array([
    [[1, 2, 3],
     [4, 5, 6]],
    [[7, 8, 9],
     [10, 11, 12]]])

import matplotlib.pyplot as plt
people = ['Adam Chase','Ben Doyle','Sam Denby','Toby Hendy']
y_pos = np.arange(len(people))
number_of_wins = (7,7,6,2)
fig, ax = plt.subplots(figsize=(12,5), facecolor='moccasin')
ax.barh(people,number_of_wins,label=people,color=['gold','slategrey','peru','orangered'])
ax.set_yticks(y_pos, labels=[])
ax.invert_yaxis()
ax.tick_params(axis='y',length=0)
ax.set_xlabel('Number of Wins',fontsize='14')
ax.set_title('Jet Lag the Game Leaderboard',fontweight='bold',fontsize='18')
ax.set_ylabel('Contestants',fontsize='14')
ax.legend()
plt.show()
fig.savefig('Jet Lag the Game Leaderboard.pdf')