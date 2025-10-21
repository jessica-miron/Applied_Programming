# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 20:41:02 2025

@author: jcmir
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

board = [['O', '.', '.'],
         ['.', 'X', 'X'],
         ['O', '.', '.']]
board[1][0] = 'X'