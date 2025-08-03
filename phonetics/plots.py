import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation, PillowWriter

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


def plot_static_trajectory(df_phrase, 
                           df_anotated, 
                           ax = None,
                           xlim = None,
                           ylim = None):

    #print(make_string_phrase(df_phrase))
    phones = df_phrase['phone_base'].unique()

    if ax is None:
        fig = plt.figure(figsize=(10,6)) 
        ax = fig.add_subplot(111)
        
    ax.scatter(df_anotated['x'],
            df_anotated['y'],
            alpha=0.05,
            s = 0.05, 
            color= 'grey')

    for ph in phones:
        mask = df_anotated['phone_base'] == ph
        df_filter = df_anotated[mask]
        ax.scatter(
        df_filter['x'],
        df_filter['y'],
        alpha=0.1,
        s = 0.05, 
        label = ph)

        x_cm, y_cm = df_filter[['x', 'y']].median()
        ax.annotate(
        ph,
        xy=(x_cm, y_cm),
        xytext=(x_cm, y_cm),  # Offset for the text position
        arrowprops=dict(arrowstyle="->", color='gray'),
        fontsize=14,
        fontweight='bold',
        color='black'
        )

        ax.scatter(df_phrase['x'],
                df_phrase['y'],
                alpha=0.7,
                s = 10.0, 
                color= 'red')
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)


def make_string_phrase(df_phrase):
    s = df_phrase['phone_base']
    #result = s[s != s.shift()].reset_index(drop=True)
    string = ''.join(s[s != s.shift()])

    string = string.replace('AP', '  AP  \n').replace('SP', ' --- SP ---')

    return string


def make_phrase_animation(df_phrase, df_anotated, fps = 20, figsize = (10, 6)):

    com_df = df_anotated.groupby('phone_base')[['x', 'y']].median().rename(columns={'x': 'com_x', 'y': 'com_y'})
    df_phrase_cm = df_phrase.merge(com_df, left_on='phone_base', right_index=True)

    phones = df_phrase['phone_base'].unique()

    # Assuming dfembed_song has 'x' and 'y'
    x = df_phrase_cm['x'].values
    y = df_phrase_cm['y'].values

    x_cm = df_phrase_cm['com_x'].values
    y_cm = df_phrase_cm['com_y'].values

    phone_traj = df_phrase['phone_base'].values
    num_points = len(x)

    # Parameters
    trail_length = 20
    head_size = 30
    trail_size = 5
    alpha_trail = 0.3

    # Set up 2D figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(False)

    ax.set_xlim([-4,17])
    ax.set_ylim([-8,12])
    #ax.tight_layout()
    plt.tight_layout()

    for ph in phones:
        mask = df_anotated['phone_base'] == ph
        df_filter = df_anotated[mask]
        ax.scatter(
            df_filter['x'],
            df_filter['y'],
            alpha=0.1,
            s = 0.05, 
            label = ph)

    # Initialize trail, head and annotations
    trail_scatter, = ax.plot([], [], 'o', markersize=trail_size, alpha=alpha_trail, color='red')
    head_scatter, = ax.plot([], [], 'o', markersize=head_size / 5, color='red')

    annotation = ax.annotate(
        "", 
        xy=(0, 0), 
        fontsize=14, 
        fontweight='bold', 
        color='black'
    )

    def update(frame):
        start = max(0, frame - trail_length)
        trail_x = x[start:frame]
        trail_y = y[start:frame]

        trail_scatter.set_data(trail_x, trail_y)

        if frame < len(x):
            head_scatter.set_data([x[frame]], [y[frame]])
            annotation.set_text(phone_traj[frame])
            annotation.set_position((x_cm[frame], y_cm[frame]))
        else:
            head_scatter.set_data([], [])
            #annotation.set_text(phone_traj[frame])

        return trail_scatter, head_scatter

    # Animate
    #fps = 100
    interval = 1000/fps
    ani = FuncAnimation(fig, update, frames=num_points, interval=interval, blit=False)

    ani.save("scatter2d.gif", writer=PillowWriter(fps=fps))

    # # Display the saved GIF
    # from IPython.display import Image
    # Image(filename="scatter2d.gif")