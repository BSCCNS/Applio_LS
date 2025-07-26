import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

def make_tagged_LS_plot(df,
                phones = ['a', 'e', 't', 's', 'm', 'n'],
                figsize = (10, 6),
                alpha = 0.1, 
                s = 0.1, 
                show_global = False,
                xlim = None,
                ylim = None,
                zlim = None, 
                add_legend = True,
                label_detail = True,
                save_fig = False):
    
    '''
    df projected, anotated
    '''

    #print((['x', 'y', 'z']).isin(df.columns))
    dim = check_dimensions(df)

    print(f'Making a {dim} plot')
    
    fig = plt.figure(figsize=figsize)    
    if dim == '3d':
        ax = fig.add_subplot(111, projection='3d')
        coords = ['x', 'y', 'z']
        
    elif dim == '2d':
        ax = fig.add_subplot(111)  # No projection for 2D
        coords = ['x', 'y']
        
    cols = [df[col] for col in coords]
    if show_global:
        ax.scatter(*cols,
                alpha=0.05,
                s = 0.05, 
                color= 'grey')

    legs = []
    num_classes = len(phones)
    colors = cm.get_cmap('tab20', num_classes)

    if type(phones) == list:
        classes = phones
    elif type(phones) == dict:
        classes = phones.keys() # groups

    for i, ph in enumerate(classes):

        if type(phones) == list:
            mask = df['phone_base'] == ph
        elif type(phones) == dict:
            mask = df['phone_base'].isin(phones[ph])
        
        df_filter = df[mask]

        cols_filter = [df_filter[col] for col in coords] 
        ax.scatter(
                *cols_filter,
                alpha=alpha,
                s = s, 
                color=colors(i),
                label = ph)
        
        if type(phones) == list:
            label = ph
        elif type(phones) == dict:
            #label = f'{ph} : {phones[ph]}'
            label = f'{ph}'
            if label_detail:
                label = f'{ph} : {phones[ph]}'
        
        leg = mlines.Line2D([], [], color=colors(i), marker='o', linestyle='None',
                                    markersize=6, label=label)
        legs.append(leg)
    
    if add_legend:
        ax.legend(handles=legs)


#     legend = ax.legend(
#     title='Age Group',
#     fontsize=12,             # Legend labels font size
#     title_fontsize=14,       # Legend title font size
#     loc='upper center',
#     bbox_to_anchor=(0.5, -0.15),
#     ncol=len(pivot_df.columns),  # One column per group = single line
#     frameon=False             # Optional: removes legend box
# )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    if save_fig:
        plt.axis('off')
        #fig.savefig('LS.svg', format='svg', dpi=1200)
        fig.savefig('LS.png', dpi=1200)

def check_dimensions(df):
    cols = set(df.columns)
    if {'x', 'y', 'z'}.issubset(cols):
        return '3d'
    elif {'x', 'y'}.issubset(cols):
        return '2d'
    else:
        return None

