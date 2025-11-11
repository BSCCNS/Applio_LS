import pandas as pd
import logging
from sklearn.metrics import silhouette_score, silhouette_samples

from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score


def compute_silhouette(df_anotated):
    logging.info(f'-------- Computing silhouette')

    X, y = extract_Xy(df_anotated)
    sil_score = silhouette_score(X, y, metric='cosine')
    logging.info(f'--------- sil_score {sil_score}')
    return sil_score 

def compute_MI(df_anotated, n_clusters = 50):
    logging.info(f'-------- Computing MI')

    X, y = extract_Xy(df_anotated)

    kmeans = KMeans(n_clusters = n_clusters, 
                    random_state=42)
    
    cluster_assignments = kmeans.fit_predict(X)

    mi = mutual_info_score(y, cluster_assignments)

    logging.info(f"-------- Mutual Information (MI-phone): {mi:.4f}")
    return mi

def compute_metric_for_layer(df_anotated, param_dict):
    
    K_MI = param_dict["K_MI"]
    exclude_phones = param_dict['exclude_phones_metric']

    logging.info(f'Excluding phones {exclude_phones} from metric computations')
    
    if exclude_phones is None:
        df = df_anotated
    else:
        mask = ~df_anotated["phone_base"].isin(exclude_phones)
        df = df_anotated[mask]

    sil = compute_silhouette(df)
    mi = compute_MI(df, n_clusters = K_MI)

    return {'sil': sil, 'mi': mi}

def extract_Xy(df_anotated):
    X = df_anotated.drop(columns = ['phone_base', 'song']).values
    y = df_anotated['phone_base'].values

    return X, y

def make_df_metric(metric_dict):
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    df = df.reset_index().rename(columns={'index': 'layer'})

    return df