# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:46:10 2017

This code assumes that given the original csv table of pings, it returns a 
table with incoming timestamp split into the hour and day 
Exmaple: 2017-11-10 

@author: ykhoja
"""
import pandas as pd

PT = pd.read_csv("PingedTranslators.csv")
PT_new = PT[['translator_id','request_time','response']].copy()
#PT_new = PT_new[PT_new['response'] != 'yes']

##### Yaya's code #########
# The code below extracts the hour of each request, and whether it occured on 
# a weekday or not. It uses pandas built it .dt functions for datetime data
# types
SATURDAY = 5
timestamp = pd.to_datetime(PT_new['request_time'])
hours = timestamp.dt.hour
days = timestamp.dt.weekday # returns number of day (Sat = 5, Sun = 6)
PT_new['hour'] = hours
PT_new['weekday'] = days < SATURDAY
PT_new = PT_new[['translator_id','hour','weekday','response']]