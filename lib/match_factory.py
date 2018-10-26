import json
import numpy as np
import lib.feature_calculators as feature_calculators

# CONSTANTS
NUM_PARTICIPANTS = 10
MAPID_SUMMONERS_RIFT = 11

def load_raw_data(match_id, file_type):
    """match_id is the name of the timeline file

    """

    filename = match_id
    
    if file_type == 'match':
        filename = filename.replace('timeline', 'match')
    
    with open(filename, 'r') as f:
        return json.load(f)


def valid_match(match_json):
    """checks if match is valid based on set of constraints
    
    returns True only if
    - map is Summoner's rift (mapId == 11)
    - 10 participants (i.e., 5v5)
    - 10 people playing, no computers
    """

    if match_json['mapId'] != MAPID_SUMMONERS_RIFT: return False

    if len(match_json['participants']) != NUM_PARTICIPANTS: return False
        
    current_champion_ids = feature_calculators.get_champion_ids(match_json, NUM_PARTICIPANTS)

    if len(np.unique(current_champion_ids)) < NUM_PARTICIPANTS: return False
    
    return True

def load(match_id):
    match_json = load_raw_data(match_id, 'match')
    timeline_json = load_raw_data(match_id, 'timeline')

    if not valid_match(match_json):
        return None

    return feature_calculators.calculate_all_features(match_json, timeline_json)

def transpose_matches_to_features(matches):
    """Because of how our algorithm analyzes per-feature,
    it is easier to transpose each match (set of features for a given match) into
    a set of features where each feature has a list of all match values for that feature.

    In: [ {Match[Feature] => Value} ]
    Out: { Features[Feature] => [Value]}
        Value is indexed by the input Match's index
    """
    if not matches:
        return []

    features = {}

    feature_names = matches[0].keys()

    for feature_name in feature_names:
        features[feature_name] = [
            match[feature_name]
                for match in matches
        ]

    return features