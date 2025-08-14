import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path

from textgrids import TextGrid
import re

from sklearn.metrics.pairwise import cosine_similarity

from umap import UMAP
import joblib

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import normalize

DIM = 768
DT = 0.020 #seconds

##############################################################
### Classifier
##############################################################

def phone_binary_classifier(df_anotated, ph, print_report=False):
    df_ph    = df_anotated[df_anotated['phone_base'] == ph]
    df_no_ph = df_anotated[df_anotated['phone_base'] != ph]

    X_ph    = df_ph.drop(columns=['phone_base', 'song'])
    X_no_ph = df_no_ph.drop(columns=['phone_base', 'song'])

    # Combine your AA and non-AA vectors
    X = np.vstack([X_ph, X_no_ph])
    y = np.array([1] * len(X_ph) + [0] * len(X_no_ph))

    # Split into training and test sets (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train classifier
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Print evaluation metrics
    if print_report:
        print(classification_report(y_test, y_pred))
        print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")

    auc = roc_auc_score(y_test, y_proba)

    return auc


##############################################################
### UMAP
##############################################################

def train_umap(
    df_anotated,
    exclude_phones = ['SP'],
    n_components=3, 
    n_neighbors=100, 
    min_dist=0.2,
    n_jobs = 4,
    save_model = False,
    metric = 'euclidean',
    normalize_vectors = False,
    folder = ''):

    # train umap with dataset without silence
    if exclude_phones is not None:
        mask = ~df_anotated['phone_base'].isin(exclude_phones)
        df_filter = df_anotated[mask]
    else:
        df_filter = df_anotated

    X = df_filter.drop(columns=['phone_base', 'song']).values

    if normalize_vectors:
        X = np.asarray(X, dtype=np.float32, order="C")
        X = normalize(X, norm="l2", axis=1, copy=False)

    print(f'Training UMAP with parameters n_components : {n_components}, n_neighbors {n_neighbors}, min_dist : {min_dist}')
    reducer = UMAP(n_components=n_components, 
                n_neighbors=n_neighbors, 
                min_dist=min_dist, 
                metric = metric,
                random_state=42,
                n_jobs=n_jobs)

    reducer.fit(X)

    if save_model:
        dist = str(min_dist).replace('.', 'p')
        file = f'{folder}/umap_all_songs_n{n_neighbors}_dist{dist}_{n_components}D.sav'
        print(f'Saving model to file {file}')

        os.makedirs(folder, exist_ok=True)
        joblib.dump(reducer, file)

    return reducer

##############################################################
### Parsing
##############################################################

def find_articulations(df_song, target, padding = 1):
    """
    Find indices of all contiguous blocks of `target` in the series,
    including the element before and after each block (if they exist).
    
    Parameters:
        song (pd.Dataframe): df with song (reset index)
        target (str): The character to search for.
        
    Returns:
        List[List[int]]: List of index lists for each expanded block.
    """

    series = df_song['phone_base']

    mask = series == target
    group = (mask != mask.shift()).cumsum()
    blocks = mask.groupby(group).apply(lambda x: x.index if x.all() else None).dropna()

    expanded_indices = []
    for block in blocks:
        start = max(block[0] - padding, 0) 
        end = min(block[-1] + padding, len(series) - 1) 
        expanded_indices.append(list(range(start, end + 1)))

    print(f'Detected {len(blocks)} articulations')

    articulations = [df_song.iloc[idxs] for idxs in expanded_indices]

    return articulations

def group_vowels_consonants_ap(df_anotated,
                               vowels = ['a', 'e', 'i', 'o', 'u', '3', 'w', '0','y'],
                               AP = ['AP'], 
                               SP = 'SP'):

    print(f'vowels: {vowels}')
    print(f'aspirations: {AP}')
    print(f'Short pause: {SP}')
    #vowels = ['a', 'e', 'i', 'o', 'u', '3', 'w', '0','y']
    #AP = ['AP']

    mask = (df_anotated['phone_base'] != SP)

    df_anotated = df_anotated[mask]

    df_consonant = df_anotated[~df_anotated['phone_base'].isin(vowels + AP)].copy()
    df_vowels = df_anotated[df_anotated['phone_base'].isin(vowels)].copy()
    df_ap = df_anotated[df_anotated['phone_base'].isin(AP)].copy()

    df_ap['group'] = 'AP'
    df_vowels['group'] = 'vowel'
    df_consonant['group'] = 'consonant'

    df_grouped = pd.concat([df_consonant, df_vowels, df_ap])

    ph_group = {
    'consonants': list(df_consonant['phone_base'].unique()),
    'vowels': list(df_vowels['phone_base'].unique()),
    'AP': list(df_ap['phone_base'].unique())
    }

    return df_grouped, ph_group

def make_proj_anotated_feat_df(df_anotated, 
                               umap_model,
                               save_df = False,
                               folder = '',
                               suffix = 'all_songs_'):
    
    print('Applying dimensional reduction')
    X = df_anotated.drop(columns=['phone_base', 'song']).values
    X_projected = umap_model.transform(X)

    n_neighbors = umap_model.n_neighbors
    dim = umap_model.n_components
    min_dist = umap_model.min_dist

    print(f'Reduced to {dim} dimensions')

    if dim == 2:
        cols = ['x', 'y']
    elif dim == 3:
        cols = ['x', 'y', 'z']

    df_proj = pd.DataFrame(data = X_projected, columns=cols)
    df_proj[['phone_base', 'song']] = df_anotated[['phone_base', 'song']]

    if save_df:
        dist = str(min_dist).replace('.', 'p')
        file = f'{folder}/df_proj_anotated_{suffix}umap_n{n_neighbors}_dist{dist}_{dim}D.csv'
        os.makedirs(folder, exist_ok=True)
        df_proj.to_csv(file)

    return df_proj


def make_anotated_feat_df(feat_paths, 
                          lab_paths, 
                          from_converted = False,
                          tp_algn = 'lab',
                          dataset = 'libri',
                          add_transitions = False,
                          pad_seconds = 0.0101,
                          remove_short_phones = False):
    df = pd.concat([make_single_anotated_feat_df(f, 
                                                lab_paths,
                                                from_converted = from_converted,
                                                tp_algn = tp_algn,
                                                dataset = dataset, 
                                                add_transitions = add_transitions,
                                                pad_seconds = pad_seconds,
                                                remove_short_phones = remove_short_phones) 
                                                for f in feat_paths], axis=0)
    return df.reset_index(drop=True)

def make_single_anotated_feat_df(feat_file, 
                                 lab_paths, 
                                 from_converted = False,
                                 tp_algn = 'lab',
                                 dataset = 'libri',
                                 add_transitions = False,
                                 pad_seconds = 0.0101,
                                 remove_short_phones = False):
    df_feat = df_features_from_csv_file(feat_file)
    
    song_name = get_song_name(feat_file)

    if from_converted:
        song_name = get_song_name(feat_file).split('_output')[0]

    files_match = [f for f in lab_paths if song_name == Path(f).stem]

    assert len(files_match) > 0,  f"Lab file for {song_name} not found"
    assert len(files_match) == 1, f"Too many lab files for {song_name}"

    lab_file = files_match[0]

    if tp_algn == 'lab':
        df_algn = df_alignments_from_lab_file(lab_file, 
                                              add_transitions=add_transitions,
                                              pad_seconds=pad_seconds)
                                        
    elif tp_algn == 'text_grid':
        if dataset == 'libri':
            df_algn = make_def_single_file(lab_file, phone_key_word='phones')
        elif dataset == 'gt':
            df_algn = make_def_single_file(lab_file, phone_key_word='phone')


    t_last = df_algn['end'].iloc[-1]
    check = t_last/DT - len(df_feat)

    if from_converted:
        assert check < 4, "Lengths of lab and feat files do NOT match "
    else:
        assert check < 2, "Lengths of lab and feat files do NOT match "

    df_feat_anotated = add_phone_to_feat_df(df_feat, df_algn)
    df_feat_anotated['song'] = song_name

    if remove_short_phones:
        mask = df_feat_anotated['duration'] > DT
        df_feat_anotated = df_feat_anotated[mask]

    return df_feat_anotated

def get_song_name(feat_file):
    path = Path(feat_file)
    stem = path.stem
    return stem.split('feats_')[1]

def df_features_from_csv_file(csv_file):
    df=pd.read_csv(csv_file)
    dim_cols = [str(n) for n in range(DIM)]
    df_features = df[[c for c in df.columns if c in dim_cols]]

    return df_features

def add_phone_to_feat_df(df_feat, df_algn):
    # Initialize the column 
    df_feat = df_feat.copy()
    df_feat['phone_base'] = None
    df_feat['duration'] = None

    # Assign phone_base based on start_idx and end_idx
    for _, row in df_algn.iterrows():
        df_feat.loc[row['start_idx']:row['end_idx'] - 1, 'phone_base'] = row['phone_base']
        df_feat.loc[row['start_idx']:row['end_idx'] - 1, 'duration'] = row['duration']

    return df_feat

###########################################################################
###### songs
###########################################################################

def df_alignments_from_lab_file(lab_file, 
                                add_transitions = False,
                                pad_seconds=0.010):
    '''
    This uses the conventions of the lab files Maria gave us
    '''
    
    df = pd.read_csv(
        lab_file, 
        sep='\s+',
        header=None, 
        names=["start", "end", "label"]
    )

    dt = DT #in seconds

    df = df.rename(columns={'label': 'phone_base'})
    df[['start', 'end']] = df[['start', 'end']]/1e7
    df['duration'] = df['end'] - df['start']

    if add_transitions:
        df = insert_transitions(df, 
                                pad_seconds=pad_seconds, 
                                transition_label="transition")
        
        df = df[df['duration'] < DT]

    df["start_idx"] =  (df["start"]/dt).apply(np.floor).astype(int)
    df["end_idx"] =  (df["end"]/dt).apply(np.floor).astype(int)

    return df

def insert_transitions(df, pad_seconds=0.010, transition_label="transition"):
    import pandas as pd
    if df.empty:
        return df.copy()
    df_sorted = df.sort_values(["start", "end"]).reset_index(drop=True)
    out_rows = []
    current = df_sorted.iloc[0].to_dict()

    for i in range(len(df_sorted) - 1):
        left = current
        right = df_sorted.iloc[i + 1].to_dict()

        left_len = max(0.0, float(left["end"]) - float(left["start"]))
        right_len = max(0.0, float(right["end"]) - float(right["start"]))
        shave_left = min(pad_seconds, left_len)
        shave_right = min(pad_seconds, right_len)

        t_start = float(left["end"]) - shave_left
        t_end = float(right["start"]) + shave_right

        if t_end < t_start:
            mid = (float(left["end"]) + float(right["start"])) / 2.0
            t_start = mid
            t_end = mid
            shave_left = max(0.0, min(float(left["end"]) - t_start, left_len))
            shave_right = max(0.0, min(t_end - float(right["start"]), right_len))

        trimmed_left = dict(left)
        trimmed_left["end"] = max(trimmed_left["start"], t_start)
        out_rows.append(trimmed_left)

        out_rows.append({"start": t_start, "end": t_end, "phone_base": transition_label})

        right["start"] = min(float(right["end"]), t_end)
        current = right

    out_rows.append(current)
    return pd.DataFrame(out_rows).reset_index(drop=True)

###########################################################################
###### libri speech
###########################################################################

def make_def_single_file(path, phone_key_word = "phones"):
    
    tg = TextGrid()
    tg.read(path)

    #phones_tier = tg["phones"]
    phones_tier = tg[phone_key_word]

    phones_df = pd.DataFrame([
        {
            "start": interval.xmin,
            "end": interval.xmax,
            "duration": interval.xmax - interval.xmin,
            "phone": interval.text
        }
        for interval in phones_tier
    ])

    phones_df["phone_base"] = phones_df["phone"].apply(strip_stress)
    phones_df["start_idx"] =  (phones_df["start"]/0.02).apply(np.floor).astype(int)
    phones_df["end_idx"] =  (phones_df["end"]/0.02).apply(np.floor).astype(int)

    return phones_df

def strip_stress(phoneme):
    return re.sub(r'\d$', '', phoneme)


##############################################################
### Metrics
##############################################################

def similarity(X, Y, out_time = False):

    if len(X.shape) == 1:
        X = X.values.reshape(1,-1)

    if len(Y.shape) == 1:
        Y = Y.values.reshape(1,-1)

    if out_time:
        dt = 0.02
        N = max(len(X),len(Y))
        t = dt*np.array(range(N))
        return t, cosine_similarity(X, Y)[0]
    else:
        return cosine_similarity(X, Y)[0]
    
def phone_correlation(df, phone = 't'):

    df = df.reset_index(drop = True)

    df_phone = df[df['phone_base'] == phone]
    drop_cols = ['phone_base', 'song', 'speaker', 'book', 'line']
    template = df_phone.drop(columns=drop_cols).mean()

    stream = df.drop(columns=drop_cols)

    corr = similarity(template, stream.values)

    matches = df_phone.index

    return corr, matches