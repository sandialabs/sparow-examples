from pathlib import Path
from pyomo.core import TransformationFactory
from .gtep_model import ExpansionPlanningModel
from .gtep_data import ExpansionPlanningData

current_file_dir = Path(__file__).resolve().parent


def create_gtep_model(
    *,
    num_stages,
    num_rep_days,
    len_rep_days,
    num_commit_p,
    num_disp,
    alpha=1.0,
    flow_model="CP",
    include_commitment=True,
    representative_dates=None,
    representative_weights=None,
):
    if representative_dates is None:
        representative_dates = [
        '2020-01-28 00:00',
        '2020-04-23 00:00',
        '2020-07-05 00:00',
        '2020-10-14 00:00'
    ]
    if representative_weights is None:
        representative_weights = [1, 1, 1, 1]

    data_path = str(current_file_dir / "data")
    data_object = ExpansionPlanningData(
        stages=num_stages,
        num_reps=num_rep_days,
        len_reps=len_rep_days,
        num_commit=num_commit_p,
        num_dispatch=num_disp,
    )

    data_object.load_prescient(
        data_path,
        representative_dates=representative_dates,
        representative_weights=representative_weights,
    )

    mod_object = ExpansionPlanningModel(
        data=data_object,
    )

    mod_object.config["include_commitment"] = include_commitment
    #mod_object.config["alpha_scaler"] = alpha
    mod_object.config["flow_model"] = flow_model
    mod_object.config["storage"] = True
    mod_object.config["transmission"] = True
    mod_object.config["thermal_generation"] = True
    mod_object.config["renewable_generation"] = True
    mod_object.config["scale_loads"] = False
    mod_object.config["scale_texas_loads"] = False

    mod_object.create_model()
    TransformationFactory("gdp.bound_pretransformation").apply_to(mod_object.model)
    TransformationFactory("gdp.bigm").apply_to(mod_object.model)

    return mod_object.model
