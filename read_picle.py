import pickle
import pandas as pd

# 불러오기
# with open('marinedebrisdetector/checkpoints/1/unet++2/epoch=54-val_loss=0.50-auroc=0.987 (copy)/archive/data.pkl', 'rb') as f:
#     data = pickle.load(f)
dataframe = pd.read_pickle('marinedebrisdetector/checkpoints/1/unet++2/epoch=54-val_loss=0.50-auroc=0.987 (copy)/archive/data.pkl')
