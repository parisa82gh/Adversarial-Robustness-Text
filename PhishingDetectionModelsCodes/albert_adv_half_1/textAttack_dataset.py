# Import csv data into textattack format
import textattack
import pandas as pd

# Create a dataframe from csv
df = pd.read_csv('/project/verma/TextAttack_Parisa/adv_data/test_f_adv_half.csv', delimiter=',')
#df = df.iloc[:100]
# Create a list of tuples for Dataframe rows using list comprehension
list_of_tuples = [tuple(row) for row in df.values]


dataset = textattack.datasets.Dataset(list_of_tuples)

