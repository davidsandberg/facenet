# Automatically generated by Pynguin.
import download_and_extract as module_0


def test_case_0():
    str_0 = (
        'Model definition for the variational autoencoder. Points to a module containing the definition.'
        )
    var_0 = module_0.download_file_from_google_drive(str_0, str_0)
    assert var_0 is None
    assert module_0.model_dict == {'lfw-subset':
        '1B5BQUZuJO-paxdN8UclxeHAR1WnR_Tzi', '20170131-234652':
        '0B5MzpY9kBtDVSGM0RmVET2EwVEk', '20170216-091149':
        '0B5MzpY9kBtDVTGZjcWkzT3pldDA', '20170512-110547':
        '0B5MzpY9kBtDVZ2RpVDYwWmxoSUk', '20180402-114759':
        '1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-'}
