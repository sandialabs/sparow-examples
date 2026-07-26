import json
import or_topas.aos
from sparow_examples.aos_gtep_9bus.aos_single import create_sp
from sparow.sp.util import relax_second_stage

OUTPUT_FILE = 'aos_single_scenario_ef_hamming.json'


print("\n--- Creating model ---")
sp = create_sp()
sp.add_transformation(relax_second_stage)
model = sp.create_EF(compact_repn=True)

print("\n--- Running AOS ---")
# aos_results = or_topas.aos.enumerate_binary_solutions(model, num_solutions=2, solver="gurobi")
rel_opt_gap = 0.00001
sol_max = 3
search_mode='hamming'
variables = None
aos_results = or_topas.aos.enumerate_binary_solutions(model, 
                                                      variables=variables, 
                                                      num_solutions=sol_max, 
                                                      rel_opt_gap=rel_opt_gap, 
                                                      solver="gurobi",
                                                      search_mode=search_mode)

if True:
    print("\n--- Results Summary ---")
    print(f"Metadata: Max Sols {sol_max}, Rel_Gap {rel_opt_gap}")
    print(f"Number of Solutions {len(aos_results)}")
    for i,s in enumerate(aos_results):
        print(f"Sol {i}: objective: {s.objective().value}")

if False:
    print("\n--- Printing results ---")
    for s in aos_results:
        print(s)
        print(s.objective().value)

print("\n--- Saving results to an output file ---")
aos_dict = aos_results.to_dict()
with open(OUTPUT_FILE,'w') as OUTPUT:
    json.dump(aos_dict, OUTPUT, indent=0)

