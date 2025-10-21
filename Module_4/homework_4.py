# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 10:34:46 2025

@author: jcmir
"""

import pandas as pd

def createDataframe(student_data: List[List[int]]) -> pd.DataFrame:
    return pd.DataFrame(student_data, columns=['student_id','age'])

def getDataframeSize(players: pd.DataFrame) -> List[int]:
    shape = players.shape
    rows, columns = shape
    return [rows,columns]

def selectFirstRows(employees: pd.DataFrame) -> pd.DataFrame:
    return employees.head(3)

def selectData(students: pd.DataFrame) -> pd.DataFrame:
    student = students.loc[students['student_id']==101]
    answer = student.loc[:,['name','age']]
    return answer

def createBonusColumn(employees: pd.DataFrame) -> pd.DataFrame:
    employees['bonus'] = employees['salary'] * 2
    return employees

def dropDuplicateEmails(customers: pd.DataFrame) -> pd.DataFrame:
    return customers.drop_duplicates(subset='email')

def dropMissingData(students: pd.DataFrame) -> pd.DataFrame:
    return students.dropna(subset='name')

def modifySalaryColumn(employees: pd.DataFrame) -> pd.DataFrame:
    employees['salary'] = employees['salary'] *2
    return employees

def renameColumns(students: pd.DataFrame) -> pd.DataFrame:
    return students.rename(columns={'id':'student_id','first':'first_name','last':'last_name','age':'age_in_years'})

def changeDatatype(students: pd.DataFrame) -> pd.DataFrame:
    return students.astype({'grade':int})

def fillMissingValues(products: pd.DataFrame) -> pd.DataFrame:
    products['quantity'].fillna(0,inplace=True)
    return products

def concatenateTables(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df1,df2])

def pivotTable(weather: pd.DataFrame) -> pd.DataFrame:
    return weather.pivot(index='month',columns='city',values='temperature')

def meltTable(report: pd.DataFrame) -> pd.DataFrame:
    return pd.melt(report,id_vars='product',value_vars=['quarter_1','quarter_2','quarter_3','quarter_4'],var_name='quarter',value_name='sales')

def findHeavyAnimals(animals: pd.DataFrame) -> pd.DataFrame:
    heavy_animals = animals.loc[animals['weight']>100].sort_values(by='weight',ascending=False)
    return heavy_animals.loc[:,['name']]