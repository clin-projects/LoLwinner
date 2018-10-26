import glob

def log(LOG_LEVEL, print_statement):
	# TODO: I don't need real log levels for now, just need off / on
	if LOG_LEVEL == 'Debug': print(print_statement)

def get_match_ids(folder):
	return glob.glob(folder + '/*timeline*')