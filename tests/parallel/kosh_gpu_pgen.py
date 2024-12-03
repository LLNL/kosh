from maestrowf.datastructures.core import ParameterGenerator

NUM_STUDIES = 2  # 1024
DATASETS = 2  # 1024
ENSEMBLES = 2  # 1024


def get_custom_generator(env, **kwargs):

    p_gen = ParameterGenerator()

    params = {"RUN_NUMBER": {"values": list(range(NUM_STUDIES)),
                             "label": "RUN_NUMBER.%%"},

              "DATASETS": {"values": [DATASETS]*NUM_STUDIES,
                           "label": "DATASETS.%%"},

              "ENSEMBLES": {"values": [ENSEMBLES]*NUM_STUDIES,
                            "label": "ENSEMBLES.%%"},
              }

    for key, value in params.items():
        p_gen.add_parameter(key, value["values"], value["label"])

    return p_gen
