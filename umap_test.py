import pandas as pd
import matplotlib.pyplot as plt

from phonetics import utils as u
#from phonetics import phone_info as ph_i 
from phonetics import plots as plots
from phonetics import metrics as ph_metrics


#############################################################

df_anotated = pd.read_csv("")

print(f'-------- umap')
umap2 = u.train_umap(
df_anotated,
exclude_phones = None,
n_components=2, 
n_neighbors=100, 
min_dist=0.1,
save_model = False,
folder = None)
    
df_proj_anotated = u.make_proj_anotated_feat_df(df_anotated, 
                                                umap2,
                                                save_df = False,
                                                folder = None)

def make_plot(df_proj_anotated):
    print(f'-------- plot')
    my_phones = [k for k in df_proj_anotated['phone_base'].value_counts().keys() if k != 'AP']
    plots.make_tagged_LS_plot(df_proj_anotated,
            phones = my_phones,
            alpha = 0.25, 
            s = 0.1,
            show_global=True)
    plt.savefig(f'LS_layer_12_test')

