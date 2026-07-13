#
# Define the create_gtep_model() function
#
from pathlib import Path
from pyomo.core import TransformationFactory
from gtep.gtep_model import ExpansionPlanningModel
from gtep.gtep_data import ExpansionPlanningData

current_file_dir = Path(__file__).resolve().parent


def create_gtep_model(
    *, num_stages, num_rep_days, len_rep_days, num_commit_p, num_disp, alpha=1.0
):
    data_path = current_file_dir / "data"
    data_object = ExpansionPlanningData()
    data_object.load_prescient(data_path)
    # data_object.load_storage_csv(data_path)

    mod_object = ExpansionPlanningModel(
        stages=num_stages,
        data=data_object,
        num_reps=num_rep_days,  # num rep days
        len_reps=len_rep_days,  # len rep days
        num_commit=num_commit_p,  # num commitment periods
        num_dispatch=num_disp,  # num dispatch per commitment period
    )

    mod_object.config["include_commitment"] = True
    mod_object.config["alpha_scaler"] = alpha
    mod_object.config["flow_model"] = "CP"  # change this to "DC" to run DCOPF!
    mod_object.config["storage"] = True
    mod_object.config["transmission"] = True  # TRANSMISSION INVESTMENT FLAG
    mod_object.config["thermal_generation"] = True  # THERMAL GENERATION INVESTMENT FLAG
    mod_object.config["renewable_generation"] = (
        True  # RENEWABLE GENERATION INVESTMENT FLAG
    )
    mod_object.config["scale_loads"] = False  # LEAVE AS FALSE
    mod_object.config["scale_texas_loads"] = False  # LEAVE AS FALSE

    mod_object.create_model()
    TransformationFactory("gdp.bound_pretransformation").apply_to(mod_object.model)
    TransformationFactory("gdp.bigm").apply_to(mod_object.model)

    return mod_object.model
