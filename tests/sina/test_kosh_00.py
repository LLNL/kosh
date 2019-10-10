import sys
import os
sys.path.append(os.getcwd())
from  kosh import KoshStore
store = KoshStore(engine="sina", username="cdoutrix", sql='sql', db_path='sina.sql')
