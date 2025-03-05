def get_model_parms(name: str):
    
    hyperparameters = {
        "hyperparameters_resnet_lstm": {
            'lstm_hidden_dim': 32,
            'num_lstm_layers': 2,
            'dropout': 0.4,
            'bidirectional': False,
            'freeze': False,
        },
        "hyperparameters_eff_b3_lstm": {
            'lstm_hidden_dim': 32,
            'num_lstm_layers': 2,
            'dropout': 0.4,
            'bidirectional': False,
        },
        "hyperparameters_resnet3d": {
            'dropout': 0.4,
            'freeze': False,
        }
    }

    return hyperparameters.get(name, None)


def get_train_hparms(name: str):
    hyperparameters = {
        "hyperparameters_resnet_lstm": {
            'n_frames': 20,
            'batch_size': 4,
            'lr': 0.00001,
            'weight_decay': 1e-5,
            'num_epochs': 25,
            'gamma': 0.1,
            'size': (224, 224),
            'milestones': [32, 64]
        },
        "hyperparameters_eff_b3_lstm": {
            'n_frames': 20,
            'batch_size': 4,
            'lr': 0.00001,
            'weight_decay': 1e-5,
            'num_epochs': 25,
            'gamma': 0.1,
            'size': (224, 224),
            'milestones': [32, 64]
        },
        "hyperparameters_resnet3d": {
            'n_frames': 20,
            'batch_size': 8,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'num_epochs': 40,
            'size': (112, 112),
            'gamma': 0.1,
            'milestones': [16, 32]
        }
    }

    return hyperparameters.get(name, None)