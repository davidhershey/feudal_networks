scripts for executing training

# Configuration File Instructions
- the config file for the optimizer and the policy are loaded during runtime
- they are loaded based on an argument flag that is passed in
- the config files are expected to be placed in scripts/training/configs
- by default, the config file argument is empty, and is assumed to match the name of the policy that is passed in. So for example, if the policy is 'lstm', then the config name is assumed to be 'lstm_config' (note that this does not end with '.py' because it is used in an import statement).
- if an alternative config should be used, just specify the name, for example 'other_config'
- the config file itself should contain a class with members for each of the config values to be used, and the class should be called 'Config'