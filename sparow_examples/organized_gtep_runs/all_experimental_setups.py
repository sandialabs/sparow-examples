###------------------MULTIFIDELITY, MULTISCENARIO------------------###
###-------------A9-------------###

experimental_name = "exp_A9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------B9-------------###
experimental_name = "exp_B9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "DC",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------C9-------------###
experimental_name = "exp_C9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "CP"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------D9-------------###

###-------------E9-------------###

###-------------F9-------------###

###-------------G9-------------###
experimental_name = "exp_G9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.0
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}

###-------------H9-------------###
experimental_name = "exp_H9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.0
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "DC",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------I9-------------###
experimental_name = "exp_I9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.0
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "CP"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------J9-------------###
experimental_name = "exp_J9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}
###-------------K9-------------###
experimental_name = "exp_K9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "DC",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}
###-------------L9-------------###
experimental_name = "exp_L9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "CP"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}
###-------------M9-------------###

###-------------N9-------------###

###-------------O9-------------###

###-------------P9-------------###
experimental_name = "exp_P9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.0
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}
###-------------Q9-------------###
experimental_name = "exp_Q9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.0
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "DC",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}
###-------------R9-------------###
experimental_name = "exp_R9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "CP"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}


###------------------SINGLEFIDELITY, MULTISCENARIO------------------###

###-------------SFMS_A9-------------###
experimental_name = "exp_SFMS_A9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "CP"
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------SFMS_B9-------------###
experimental_name = "exp_SFMS_B9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "DC",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------SFMS_C9-------------###
experimental_name = "exp_SFMS_C9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "CP"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": True, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------SFMS_D9-------------###
experimental_name = "exp_SFMS_D9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "DC",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": True, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------SFMS_E9-------------###
experimental_name = "exp_SFMS_E9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "CP"
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}
###-------------SFMS_F9-------------###
experimental_name = "exp_SFMS_F9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "DC",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}
###-------------SFMS_G9-------------###
experimental_name = "exp_SFMS_G9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "CP"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": True, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}
###-------------SFMS_H9-------------###
experimental_name = "exp_SFMS_H9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
    "scenario_B": 4,
}

power_flow_fidelity = {
    "scenario_A": "DC",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": True, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}
###-------------SFMS_I9-------------###
experimental_name = "exp_SFMS_I9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 12,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "CP"
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------SFMS_J9-------------###
experimental_name = "exp_SFMS_J9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 12,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "DC",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------SFMS_K9-------------###
experimental_name = "exp_SFMS_K9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 12,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "CP"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": True, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------SFMS_L9-------------###
experimental_name = "exp_SFMS_L9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 12,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "DC",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": True, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 4,
    "scenario_B": 4
}
###-------------SFMS_M9-------------###
experimental_name = "exp_SFMS_M9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 12,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "CP"
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}
###-------------SFMS_N9-------------###
experimental_name = "exp_SFMS_N9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 12,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "DC",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": False, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}
###-------------SFMS_O9-------------###
experimental_name = "exp_SFMS_O9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 12,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "CP",
    "scenario_B": "CP"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": True, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}
###-------------SFMS_P9-------------###
experimental_name = "exp_SFMS_P9"
case_study = "9-bus"

scenarios = ["scenario_A", "scenario_B"]

alpha = {
    "scenario_A": 1.0,
    "scenario_B": 1.2
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 12,
    "scenario_B": 12,
}

power_flow_fidelity = {
    "scenario_A": "DC",
    "scenario_B": "DC"
}

relaxations = {
    "scenario_A": {"relax_second_stage": True, "unit_commitment": True},
    "scenario_B": {"relax_second_stage": True, "unit_commitment": True}
}

number_of_commitment = {
    "scenario_A": 12,
    "scenario_B": 12
}


###------------------SINGLEFIDELITY, SINGLESCENARIO------------------###
###-------------SF_A1_A9-------------###
experimental_name = "exp_SFSS_A9"
case_study = "9-bus"

scenarios = ["scenario_A"]

alpha = {
    "scenario_A": 1.0,
}

growth_rate = 1.00

num_representative_days = {
    "scenario_A": 4,
}

power_flow_fidelity = {
    "scenario_A": "CP",
}

relaxations = {
    "scenario_A": {"relax_second_stage": False, "unit_commitment": True},
}

number_of_commitment = {
    "scenario_A": 4,
}
###-------------SF_A1_B9-------------###
###-------------SF_A1_C9-------------###
###-------------SF_A1_D9-------------###
###-------------SF_A1_E9-------------###
###-------------SF_A1_F9-------------###
###-------------SF_A1_H9-------------###
###-------------SF_A1_I9-------------###
###-------------SF_A1_J9-------------###
###-------------SF_A1_K9-------------###
###-------------SF_A1_L9-------------###
###-------------SF_A1_M9-------------###
###-------------SF_A1_N9-------------###
###-------------SF_A1_O9-------------###
###-------------SF_A1_P9-------------###

###-------------SF_A1p2_A9-------------###
###-------------SF_A1p2_B9-------------###
###-------------SF_A1p2_C9-------------###
###-------------SF_A1p2_D9-------------###
###-------------SF_A1p2_E9-------------###
###-------------SF_A1p2_F9-------------###
###-------------SF_A1p2_H9-------------###
###-------------SF_A1p2_I9-------------###
###-------------SF_A1p2_J9-------------###
###-------------SF_A1p2_K9-------------###
###-------------SF_A1p2_L9-------------###
###-------------SF_A1p2_M9-------------###
###-------------SF_A1p2_N9-------------###
###-------------SF_A1p2_O9-------------###
###-------------SF_A1p2_P9-------------###