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



classic_season_to_doctor_mapping = {
    1: "First Doctor",
    2: "First Doctor",
    3: "First Doctor",
    4: "Second Doctor",
    5: "Second Doctor",
    6: "Second Doctor",
    7: "Third Doctor",
    8: "Third Doctor",
    9: "Third Doctor",
    10: "Third Doctor",
    11: "Third Doctor",
    12: "Fourth Doctor",
    13: "Fourth Doctor",
    14: "Fourth Doctor",
    15: "Fourth Doctor",
    16: "Fourth Doctor",
    17: "Fourth Doctor",
    18: "Fourth Doctor",
    19: "Fifth Doctor",
    20: "Fifth Doctor",
    21: "Fifth Doctor",
    22: "Sixth Doctor",
    23: "Sixth Doctor",
    24: "Seventh Doctor",
    25: "Seventh Doctor",
    26: "Seventh Doctor",
}

modern_season_to_doctor_mapping = {
    1: "Ninth Doctor",
    2: "Tenth Doctor",
    3: "Tenth Doctor",
    4: "Tenth Doctor",
    5: "Eleventh Doctor",
    6: "Eleventh Doctor",
    7: "Eleventh Doctor",
    8: "Twelfth Doctor",
    9: "Twelfth Doctor",
    10: "Twelfth Doctor",
    11: "Thirteenth Doctor",
    12: "Thirteenth Doctor",
    13: "Thirteenth Doctor",
    14: "Fifteenth Doctor",
    15: "Fifteenth Doctor",
}

doctor_to_actor_mapping = {
    "First Doctor": "William Hartnell",
    "Second Doctor": "Patrick Troughton", 
    "Third Doctor": "Jon Pertwee",
    "Fourth Doctor": "Tom Baker",
    "Fifth Doctor": "Peter Davison",
    "Sixth Doctor": "Colin Baker",
    "Seventh Doctor": "Sylvester McCoy",
    "Ninth Doctor": "Christopher Eccleston",
    "Tenth Doctor": "David Tennant", 
    "Eleventh Doctor": "Matt Smith",
    "Twelfth Doctor": "Peter Capaldi",
    "Thirteenth Doctor": "Jodie Whittaker",
    "Fifteenth Doctor": "Ncuti Gatwa",   
}