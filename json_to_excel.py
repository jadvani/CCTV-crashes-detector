# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:23:01 2020

@author: Javier
"""

import pandas as pd

df = pd.read_json(r'JSON\\983_1099.json')
df.to_excel(r'JSON\\983_1099.xlsx', index = None,sheet_name='Sheet1')

  
# Iterating through the json 
# list 

# Closing file 
# f.close() 

  