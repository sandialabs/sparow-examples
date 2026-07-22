import json
import or_topas.aos
from sparow_examples.aos_gtep_9bus.aos_single import create_sp
from sparow.sp.util import relax_second_stage

OUTPUT_FILE = 'aos_single.json'


print("\n--- Creating model ---")
sp = create_sp()
sp.add_transformation(relax_second_stage)
model = sp.create_EF(compact_repn=True)

print("\n--- Running AOS ---")
aos_results = or_topas.aos.enumerate_binary_solutions(model, num_solutions=2, solver="gurobi")

if False:
    print("\n--- Printing results ---")
    for s in aos_results:
        print(s)
        print(s.objective().value)

print("\n--- Saving results to an output file ---")
aos_dict = aos_results.to_dict()
with open(OUTPUT_FILE,'w') as OUTPUT:
    json.dump(aos_dict, OUTPUT, indent=0)

