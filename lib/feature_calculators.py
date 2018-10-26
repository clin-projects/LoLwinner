import numpy as np

# NOTATION (throughout file)
# team_0: BLUE team
# team_1: RED team

# CONSTANTS

NUM_PARTICIPANTS = 10
MAPID_SUMMONERS_RIFT = 11
TEAM_SIZE = NUM_PARTICIPANTS / 2

def calculate_team_total_and_difference(dat, num_frames):
    """calculates total for each team
    """
    team_total = np.array([np.sum(dat[:,:5],axis=1),np.sum(dat[:,5:],axis=1)]).transpose()
    team_total_diff = np.concatenate((team_total,(team_total[:,0] - team_total[:,1]).reshape((num_frames,1))),axis=1)
    return team_total_diff

def calculate_team_max_and_difference(dat, num_frames):
    """calculates max for a *single* player on each team
    """
    team_max = np.array([np.max(dat[:,:5],axis=1),np.max(dat[:,5:],axis=1)]).transpose()
    team_max_diff = np.concatenate((team_max,(team_max[:,0] - team_max[:,1]).reshape((num_frames,1))),axis=1)
    return team_max_diff

def get_champion_ids(match_json, num_participants = NUM_PARTICIPANTS):
    """returns champion id (i.e., character chosen) for each player
    """
    return [
        match_json['participants'][participant_id]['championId']
        for participant_id in range(num_participants)
    ]

def calculate_kills_by_match(match):
    """calculates kills by frame as [team_0 kills, team_1 kills, difference]
    """
    
    kills_by_frame = []
    
    for frame in match['frames']:
        team_0_kills = 0
        team_1_kills = 0
        for event in frame['events']:
            if event['type'] == 'CHAMPION_KILL':
                if event['killerId'] <= TEAM_SIZE: # BLUE team got kill
                    team_0_kills += 1 
                else: # RED team got kill
                    team_1_kills += 1
        kills_by_frame.append([team_0_kills, team_1_kills, team_0_kills - team_1_kills])
    return kills_by_frame

def calculate_buildings_by_match(match):
    """calculates buildings destroyed by frame
    """
    
    # These are ranked from furthest-to-closest from the Nexus
    map_buildings = {
        'OUTER_TURRET'    : 0,
        'INNER_TURRET'    : 1,
        'BASE_TURRET'     : 2,
        'UNDEFINED_TURRET': 3,
        'NEXUS_TURRET'    : 4
    }
    
    buildings_counts_by_frame = []
    
    for frame in match['frames']:
        cur_building_counts = np.zeros((3,5))
        for event in frame['events']:
            if event['type'] == 'BUILDING_KILL':
                cur_building_type = event['buildingType']
                cur_tower_type = event['towerType']
                cur_killer_team = 0 if event['killerId'] <= TEAM_SIZE else 1
                cur_building_counts[cur_killer_team, map_buildings[cur_tower_type]] += 1
        cur_building_counts[2,:] = cur_building_counts[0,:] - cur_building_counts[1,:]
        buildings_counts_by_frame.append(cur_building_counts)

    return np.array(buildings_counts_by_frame)

def calculate_monsters_by_match(match):
    """calculates elite monsters killed by frame
    """
    
    map_monsters = {
        'BARON_NASHOR'    : 0,
        'RIFTHERALD'      : 1,
        'AIR_DRAGON'      : 2,
        'EARTH_DRAGON'    : 3,
        'WATER_DRAGON'    : 4,
        'FIRE_DRAGON'     : 5,
        'ELDER_DRAGON'    : 6
    }
    
    monster_counts_by_frame = []
    
    for frame in match['frames']:
        cur_monster_counts = np.zeros((3,7))
        for event in frame['events']:
            if event['type'] == 'ELITE_MONSTER_KILL':
                cur_monster_type = event['monsterType']
                cur_killer_team = 0 if event['killerId'] <= TEAM_SIZE else 1
                if cur_monster_type == 'DRAGON':
                    cur_monster_type = event['monsterSubType']
                cur_monster_counts[cur_killer_team, map_monsters[cur_monster_type]] += 1
        cur_monster_counts[2,:] = cur_monster_counts[0,:] - cur_monster_counts[1,:]
        monster_counts_by_frame.append(cur_monster_counts)
    return np.array(monster_counts_by_frame)

def get_dat(match):

    current_gold = []
    total_gold = []
    xp = []
    
    for frame in match['frames']:
        frame_current_gold = []
        frame_total_gold = []
        frame_xp = []
        for i in range(NUM_PARTICIPANTS):
            participant_id = str(i + 1)
            frame_current_gold.append(frame['participantFrames'][participant_id]['currentGold'])
            frame_total_gold.append(frame['participantFrames'][participant_id]['totalGold'])
            frame_xp.append(frame['participantFrames'][participant_id]['xp'])
        current_gold.append(frame_current_gold)
        total_gold.append(frame_total_gold)
        xp.append(frame_xp)

    num_frames = len(match['frames'])
    team_current_gold = calculate_team_total_and_difference(np.array(current_gold), num_frames)
    team_total_gold = calculate_team_total_and_difference(np.array(total_gold), num_frames)
    team_xp = calculate_team_total_and_difference(np.array(xp), num_frames)
    
    team_max_current_gold = calculate_team_max_and_difference(np.array(current_gold), num_frames)
    team_max_total_gold = calculate_team_max_and_difference(np.array(total_gold), num_frames)
    team_max_xp = calculate_team_max_and_difference(np.array(xp), num_frames)
    
    return team_current_gold, team_total_gold, team_xp, team_max_current_gold, team_max_total_gold, team_max_xp, num_frames

# inhibitor is well inside base; late-game predictor
# baron spawns at 20:00
# dragon spawns at 2:30 (http://leagueoflegends.wikia.com/wiki/Dragon)
# rift herald spawns at 9:50 (http://leagueoflegends.wikia.com/wiki/Rift_Herald)
# baron spawns at 20:00 http://leagueoflegends.wikia.com/wiki/Baron_Nashor
team_stats_vars = ['firstBlood','firstTower','firstDragon','firstRiftHerald','firstInhibitor','firstBaron']

def get_team_stats(match):
    team_0_stats = np.array([1*match['teams'][0][var] for var in team_stats_vars])
    team_1_stats = np.array([1*match['teams'][1][var] for var in team_stats_vars])
    team_diff = team_0_stats - team_1_stats
    return team_diff

def pad_and_tensorize(feature_matches, max_frames, num_zeroes = 2):
    """ makes analysis more efficient for model prediction step by zero-padding features (making them same duration) and converting them to a tensor form
    """
    new_dat = []
    for feature_match in feature_matches:
        pad_width = [[0,0] for n in range(num_zeroes)]
        pad_width[0][1] = max_frames - len(feature_match)
        pad_width = tuple(tuple(x) for x in pad_width)
        new_dat.append(np.pad(feature_match, pad_width, 'constant'))
    new_dat = np.array(new_dat)
    return new_dat

def rescale_tensor(tensor, new_min = -1, new_max = 1):
    min_value = np.min(tensor)
    max_value = np.max(tensor)
    return (tensor - min_value) / (max_value - min_value) * (new_max - new_min) + new_min

def calculate_all_features(match_json, timeline_json):
    
    try:
        team_current_gold, team_total_gold, team_xp, team_max_current_gold, team_max_total_gold, team_max_xp, num_frames = get_dat(
                timeline_json)
        current_tiers = [match_json['participants'][i]['highestAchievedSeasonTier'] for i in range(NUM_PARTICIPANTS)]

    except:

        return None
    
    features = {}

    features['winners']             = (match_json['teams'][0]['win'] == 'Fail') * 1
    features['current_gold']        = team_current_gold
    features['total_gold']          = team_total_gold
    features['xp']                  = team_xp
    features['max_current_gold']    = team_max_current_gold
    features['max_total_gold']      = team_max_total_gold
    features['max_xp']              = team_max_xp
    features['num_frames']          = num_frames
    features['player_tiers']        = current_tiers
    features['duration']            = match_json['gameDuration']
    features['match_ids']           = match_json['gameId']
    features['versions']            = match_json['gameVersion']
    features['game_types']          = match_json['gameType']
    features['team_stats']          = get_team_stats(match_json)
    features['champions']           = get_champion_ids(match_json, NUM_PARTICIPANTS)
    features['kills']               = calculate_kills_by_match(timeline_json)
    features['buildings']           = calculate_buildings_by_match(timeline_json)
    features['monsters']            = calculate_monsters_by_match(timeline_json)

    return features

def calculate_tensor_features(features):

    max_frames = max(features['num_frames'])

    tensor_features = {}

    tensor_features['winners'] = np.array(features['winners'])
    tensor_features['current_gold'] = pad_and_tensorize(features['current_gold'], max_frames)
    tensor_features['total_gold'] = pad_and_tensorize(features['total_gold'], max_frames)
    tensor_features['xp'] = pad_and_tensorize(features['xp'], max_frames)
    tensor_features['max_current_gold'] = pad_and_tensorize(features['max_current_gold'], max_frames)
    tensor_features['max_total_gold'] = pad_and_tensorize(features['max_total_gold'], max_frames)
    tensor_features['max_xp'] = pad_and_tensorize(features['max_xp'], max_frames)
    tensor_features['champions'] = np.array(features['champions'])
    tensor_features['kills'] = pad_and_tensorize(features['kills'], max_frames)
    tensor_features['buildings'] = pad_and_tensorize(features['buildings'], max_frames, 3)
    tensor_features['monsters'] = pad_and_tensorize(features['monsters'], max_frames, 3)
    tensor_features['num_frames'] = np.array(features['num_frames'])
    tensor_features['team_stats'] = np.array(features['team_stats'])
    tensor_features['duration'] = np.array(features['duration'])
    tensor_features['versions'] = np.array(features['versions'])
    tensor_features['game_types'] = np.array(features['game_types'])

    map_player_tiers = {
        'UNRANKED': 0,
        'BRONZE': 1,
        'SILVER': 2,
        'GOLD': 3,
        'PLATINUM': 4,
        'DIAMOND': 5,
        'MASTER': 6,
        'CHALLENGER': 7
    }

    tensor_features['player_tiers'] = np.vectorize(map_player_tiers.get)(np.array(
        features['player_tiers']))

    return tensor_features