from maestrowf.datastructures.core import ParameterGenerator


def get_custom_generator(env, **kwargs):

    p_gen = ParameterGenerator()
    NUM_STUDIES = int(kwargs.get("NUM_STUDIES", "2"))
    NUM_RETRY = int(kwargs.get("NUM_RETRY", "10"))
    ENSEMBLES = int(kwargs.get("ENSEMBLES", "2"))
    DATASETS = int(kwargs.get("DATASETS", "2"))
    KOSH_STORE = kwargs.get("KOSH_STORE", env.find("KOSH_STORE_DEFAULT").value)

    params = {"RUN_NUMBER": {"values": list(range(NUM_STUDIES)),
                             "label": "RUN_NUMBER.%%"},

              "DATASETS": {"values": [DATASETS] * NUM_STUDIES,
                           "label": "DATASETS.%%"},

              "ENSEMBLES": {"values": [ENSEMBLES] * NUM_STUDIES,
                            "label": "ENSEMBLES.%%"},

              "KOSH_STORE": {"values": [KOSH_STORE] * NUM_STUDIES,
                             "label": "KSTORE.%%"},

              "NUM_RETRY": {"values": [NUM_RETRY] * NUM_STUDIES,
                            "label": "RETRY.%%"},
              }

    for key, value in params.items():
        p_gen.add_parameter(key, value["values"], value["label"])

    return p_gen
