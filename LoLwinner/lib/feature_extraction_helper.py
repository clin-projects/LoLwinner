import numpy as np

def get_champion_ids(match_json, num_participants = 10):
    return [
        match_json['participants'][participant_id]['championId']
        for participant_id in range(num_participants)
    ]

def valid_match(match_json):
    """checks if match is valid based on list of constraints
    
    returns True only if
    - map is Summoner's rift (mapId == 11)
    - 10 participants (i.e., 5v5)
    - 10 people playing, no computers
    """
    
    if match_json['mapId'] != 11: return False

    if len(match_json['participants']) != 10: return False
        
    current_champion_ids = get_champion_ids(match_json, 10)

    if len(np.unique(current_champion_ids)) < 10: return False
    
    return True

def calculate_team_total_and_difference(dat, frames):
    team_total = np.array([np.sum(dat[:,:5],axis=1),np.sum(dat[:,5:],axis=1)]).transpose()
    team_total_diff = np.concatenate((team_total,(team_total[:,0] - team_total[:,1]).reshape((frames,1))),axis=1)
    return team_total_diff

def calculate_team_max_and_difference(dat, frames):
    team_max = np.array([np.max(dat[:,:5],axis=1),np.max(dat[:,5:],axis=1)]).transpose()
    team_max_diff = np.concatenate((team_max,(team_max[:,0] - team_max[:,1]).reshape((frames,1))),axis=1)
    return team_max_diff

def get_kills_by_match(match):
    """needs timeline file, and outputs kills by frame as [team_0 kills, team_1 kills, difference]
    """
    num_frames = len(match['frames'])
    
    kills_by_frame = []
    
    for i in range(num_frames):
        num_events = len(match['frames'][i]['events'])
        team_0_kills = 0
        team_1_kills = 0
        for j in range(num_events):
            cur_event = match['frames'][i]['events'][j]
            cur_event_type = cur_event['type']
            if cur_event_type == 'CHAMPION_KILL':
                if cur_event['killerId'] <= 5:
                    team_0_kills += 1 
                else:
                    team_1_kills += 1
                #print(cur_killer_team)
        kills_by_frame.append([team_0_kills, team_1_kills, team_0_kills - team_1_kills])
        
    return kills_by_frame

def get_buildings_by_match(match):
    """counts of buildings destroyed by frame
    """
    
    # These are ranked from furthest-to-closest from the Nexus
    map_buildings = {
        'OUTER_TURRET'    : 0,
        'INNER_TURRET'    : 1,
        'BASE_TURRET'     : 2,
        'UNDEFINED_TURRET': 3,
        'NEXUS_TURRET'    : 4
    }
    
    building_counts = []
    
    num_frames = len(match['frames'])
    
    buildings_counts_by_frame = []
    
    for i in range(num_frames):
        num_events = len(match['frames'][i]['events'])
        team_0_kills = 0
        team_1_kills = 0
        cur_building_counts = np.zeros((3,5))
        for j in range(num_events):
            cur_event = match['frames'][i]['events'][j]
            cur_event_type = cur_event['type']
            if cur_event_type == 'BUILDING_KILL':
                cur_building_type = cur_event['buildingType']
                cur_tower_type = cur_event['towerType']
                cur_killer_team = 0 if cur_event['killerId'] <= 5 else 1
                cur_building_counts[cur_killer_team,map_buildings[cur_tower_type]] += 1
        cur_building_counts[2,:] = cur_building_counts[0,:] - cur_building_counts[1,:]
        buildings_counts_by_frame.append(cur_building_counts)
    return np.array(buildings_counts_by_frame)

def get_monsters_by_match(match):
    """counts of monsters killed by frame
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
    
    monster_counts = []
    
    num_frames = len(match['frames'])
    
    monster_counts_by_frame = []
    
    for i in range(num_frames):
        num_events = len(match['frames'][i]['events'])
        cur_monster_counts = np.zeros((3,7))
        for j in range(num_events):
            cur_event = match['frames'][i]['events'][j]
            cur_event_type = cur_event['type']
            if cur_event_type == 'ELITE_MONSTER_KILL':
                cur_monster_type = cur_event['monsterType']
                cur_killer_team = 0 if cur_event['killerId'] <= 5 else 1
                if cur_monster_type == 'DRAGON':
                    cur_monster_type = cur_event['monsterSubType']
                cur_monster_counts[cur_killer_team, map_monsters[cur_monster_type]] += 1
        cur_monster_counts[2,:] = cur_monster_counts[0,:] - cur_monster_counts[1,:]
        monster_counts_by_frame.append(cur_monster_counts)
    return np.array(monster_counts_by_frame)

def get_dat(match):
    current_gold = []
    total_gold = []
    xp = []
    frames = len(match['frames'])
    
    for frame in range(frames):
        frame_current_gold = []
        frame_total_gold = []
        frame_xp = []
        for i in range(1,11):
            frame_current_gold.append(match['frames'][frame]['participantFrames'][str(i)]['currentGold'])
            frame_total_gold.append(match['frames'][frame]['participantFrames'][str(i)]['totalGold'])
            frame_xp.append(match['frames'][frame]['participantFrames'][str(i)]['xp'])
        current_gold.append(frame_current_gold)
        total_gold.append(frame_total_gold)
        xp.append(frame_xp)
        
    team_current_gold = calculate_team_total_and_difference(np.array(current_gold), frames)
    team_total_gold = calculate_team_total_and_difference(np.array(total_gold), frames)
    team_xp = calculate_team_total_and_difference(np.array(xp), frames)
    
    team_max_current_gold = calculate_team_max_and_difference(np.array(current_gold), frames)
    team_max_total_gold = calculate_team_max_and_difference(np.array(total_gold), frames)
    team_max_xp = calculate_team_max_and_difference(np.array(xp), frames)
    
    return team_current_gold, team_total_gold, team_xp, team_max_current_gold, team_max_total_gold, team_max_xp, frames

team_stats_vars = ['firstBlood','firstTower','firstDragon','firstRiftHerald','firstInhibitor','firstBaron']

# inhibitor is well inside base; late-game predictor
# baron spawns at 20:00
# dragon spawns at 2:30 (http://leagueoflegends.wikia.com/wiki/Dragon)
# rift herald spawns at 9:50 (http://leagueoflegends.wikia.com/wiki/Rift_Herald)
# baron spawns at 20:00 http://leagueoflegends.wikia.com/wiki/Baron_Nashor

def get_team_stats(match):
    team_0_stats = np.array([1*match['teams'][0][var] for var in team_stats_vars])
    team_1_stats = np.array([1*match['teams'][1][var] for var in team_stats_vars])
    team_diff = team_0_stats - team_1_stats
    return team_diff

def convert_to_tensor(dat, num_matches, max_frames, num_zeroes = 2):
    new_dat = []
    for i in range(num_matches):
        num_frames = len(dat[i])
        pad_width = [[0,0] for n in range(num_zeroes)]
        pad_width[0][1] = max_frames - num_frames
        pad_width = tuple(tuple(x) for x in pad_width)
#        new_dat.append(np.pad(dat[i], ((0,max_frames - num_frames[i]), tuple([0]*num_zeroes)), 'constant'))
        new_dat.append(np.pad(dat[i], pad_width, 'constant'))
    new_dat = np.array(new_dat)
    return new_dat