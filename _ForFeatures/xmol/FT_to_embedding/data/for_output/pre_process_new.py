import pandas as pd
from pathlib import Path
import os 
import sys

prj_path = Path(__file__).parent.resolve()

data_dir = sys.argv[1]
datatype = sys.argv[2]


print('data type: {}'.format(datatype))
csv_dir = os.path.join(str(data_dir), str(datatype)+"_drugs.csv")

cpi = pd.read_csv(csv_dir)
cpi.reset_index(inplace=True)

max_length = 0
over512_count = 0

for i, (smile,index,drugid) in cpi[['smiles','index','drugid']].iterrows():
    print('current running drug: {}'.format(drugid))
    save_path = prj_path / str(datatype) / str(drugid)
    try:
        save_path.mkdir(parents=True)
    except:pass
    print("Warning! type of {}'s smile is {}".format(drugid,type(smile))) if type(smile) is not str else None
    smile = str(smile)
    pd.DataFrame([smile,index]).T.to_csv(os.path.join(str(save_path), str(drugid)+".tsv"),sep='\t',index=False,header=['text_a','label'])
    if len(smile)>max_length:
        max_length = len(smile)
    over512_count += 1 if len(smile)>511 else 0

print("max_length of smiles in " + datatype + ": " + str(max_length))
print("over512_count of smiles in " + datatype + ": " + str(over512_count))