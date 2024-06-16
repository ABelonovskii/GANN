import yaml

class Parameters():

    with open("PARAMETERS/config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    NUMBER_OF_CANDLES = config['NUMBER_OF_CANDLES']
    data_name = config['data_name']
    number_of_pieces = config['number_of_pieces']
    bot_name = config['bot_name']
    regime = config['mode']

    load_best = config['evolution_settings']['load_best']
    if load_best == 1:
        name_best = config['evolution_settings']['name_best']
    else:
        name_best = ''
    load_instance = 0
    name_instance = ''
    num_generations = config['evolution_settings']['num_generations']
    num_bot = config['evolution_settings']['num_bot']
    num_parents_mating = config['evolution_settings']['num_parents_mating']
    parent_selection_type = config['evolution_settings']['parent_selection_type']
    crossover_type = config['evolution_settings']['crossover_type']
    mutation_type = config['evolution_settings']['mutation_type']
    mutation_percent_genes = config['evolution_settings']['mutation_percent_genes']
    keep_parents = config['evolution_settings']['keep_parents']

    parameters_GA = {'num_generations': num_generations,
                     'num_bot': num_bot,
                     'num_parents_mating': num_parents_mating,
                     'parent_selection_type': parent_selection_type,
                     'crossover_type': crossover_type,
                     'mutation_type': mutation_type,
                     'mutation_percent_genes': mutation_percent_genes,
                     'keep_parents': keep_parents}

    number_of_piece = config['use_settings']['number_of_piece']
    build_plot = config['use_settings']['build_plot']
    name_best_use = config['use_settings']['name_best_use']

