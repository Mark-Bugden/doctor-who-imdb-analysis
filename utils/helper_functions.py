import pandas as pd

def prepare_contributor_data(df, column, min_episodes=3):
    """
    Explodes the given column (e.g. 'writers' or 'directors') in the DataFrame,
    filters out contributors with fewer than `min_episodes`,
    and assigns fractional weights for modeling.

    Returns a DataFrame with: rating, doctor, contributor, role, weight
    """
    # Explode the column
    exploded = df[['rating', 'doctor', column]].copy()
    exploded[column] = exploded[column].str.split(',').apply(lambda x: [i.strip() for i in x])
    exploded = exploded.explode(column).rename(columns={column: 'contributor'})
    
    # Count contributions
    contributor_counts = exploded['contributor'].value_counts()
    valid_contributors = contributor_counts[contributor_counts >= min_episodes].index

    # Filter contributors
    exploded = exploded[exploded['contributor'].isin(valid_contributors)].copy()

    # Compute weights per episode
    contrib_per_episode = exploded.groupby(exploded.index).contributor.transform('count')
    exploded['weight'] = 1 / contrib_per_episode
    exploded['role'] = column[:-1]  # 'writers' -> 'writer', 'directors' -> 'director'

    return exploded[['rating', 'doctor', 'contributor', 'role', 'weight']]

