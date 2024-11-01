import numpy as np
import pandas as pd

def py_func (sasdate:float):
	# complicated python logic
	ser = pd.Series([sasdate])
	ser = pd.to_timedelta(ser, unit='D') + pd.Timestamp('1960-1-1')
	dttm=ser[0]
	return dttm

sasdate = 19778
print('The SAS date ', sasdate, 'resolves to: ', py_func(19778.0))