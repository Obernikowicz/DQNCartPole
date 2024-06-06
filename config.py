import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    learning_rate = float(config['learning_rate'])
    batch_size = int(config['batch_size'])
    discount_factor = float(config['discount_factor'])
    eps_start = float(config['eps_start'])
    eps_end = float(config['eps_end'])
    eps_decay = float(config['eps_decay'])
    update_rate = float(config['update_rate'])
    episodes = int(config['episodes'])

    return (learning_rate,
            batch_size,
            discount_factor,
            eps_start,
            eps_end,
            eps_decay,
            update_rate,
            episodes)
