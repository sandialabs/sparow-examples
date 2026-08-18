import json
from pyomo.environ import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import pi
import os


#
# Add Data to Pyomo function
#
def add_data_to_pyomo(m,bus_df,line_df,gen_df):
    """
    Adds grid data to the model m.
    """
    
    # sets
    m.Lines = Set(initialize=line_df['UID'].tolist())
    m.Generators = Set(initialize=gen_df['GENUID'].tolist())
    m.Buses = Set(initialize=bus_df['BusID'].tolist())
    m.Loads = Set(initialize=bus_df.loc[bus_df['MW Load'] > 0, 'BusID'].tolist())


    lines = line_df['UID'].tolist()
    buses = bus_df['BusID'].tolist()
    generators = gen_df['GENUID'].tolist()
    
    demand = dict.fromkeys(buses)
    generators_in_b = dict.fromkeys(buses)
    lines_to_b = dict.fromkeys(buses)
    lines_from_b = dict.fromkeys(buses)
    total_load = 0
    for idx, row in bus_df.iterrows():
        b = int(row['BusID'])
        demand[b] = row['MW Load']
        total_load += demand[b]
        generators_in_b[b] = []
        lines_to_b[b] = []
        lines_from_b[b] = []
    generator_lb = dict.fromkeys(generators)
    generator_ub = dict.fromkeys(generators)
    generation_cost = dict.fromkeys(generators)
    bus_containing = dict.fromkeys(generators)
    for idx, row in gen_df.iterrows():
        g = row['GENUID']
        b = int(row['Bus ID'])
        generators_in_b[b].append(g)
        bus_containing[g] = b
        generator_lb[g] = float(row['PMin MW'])
        generator_ub[g] = float(row['PMax MW'])
        # just choosing this column to do linear cost atm
        #generation_cost[g] = float(row['1'])

    bus_l_to = dict.fromkeys(lines)
    bus_l_from = dict.fromkeys(lines)
    susceptance = dict.fromkeys(lines)
    thermal_limit = dict.fromkeys(lines)
    for idx, row in line_df.iterrows():
        l = row['UID']
        lines_from_b[int(row['From Bus'])].append(l)
        lines_to_b[int(row['To Bus'])].append(l)
        bus_l_to[l] = row['To Bus']
        bus_l_from[l] = row['From Bus']
        susceptance[l] = abs(1/float(row['X']))
        flowUB = float(row['LTE Rating'])
        # 0 means there are no thermal limits. We set them to total load in the
        # system since that will mean the same thing.
        thermal_limit[l] = flowUB if flowUB != 0 else total_load
    m.GeneratorsAtBus = Set(m.Buses, initialize=generators_in_b,
                            within=m.Generators)
    m.LinesFrom = Set(m.Buses, initialize=lines_from_b)
    m.LinesTo = Set(m.Buses, initialize=lines_to_b)
    # Create a mapping for LinesFrom and LinesTo
    #lines_from_b = line_df['From Bus'].tolist()
    #lines_to_b = line_df['To Bus'].tolist()

    # Create a mapping for GeneratorsAtBus
    #generators_in_b = {bus_id: [] for bus_id in bus_df['BusID']}  # Initialize all buses with empty lists
    #for index, row in gen_df.iterrows():
    #    bus_id = row['Bus ID']
    #    gen_uid = row['GENUID']
    #    if bus_id in generators_in_b:
    #        generators_in_b[bus_id].append(gen_uid)
 
    # Initialize GeneratorsAtBus set
    #m.GeneratorsAtBus = Set(m.Buses, initialize=lambda b: generators_in_b[b], within=m.Generators)
    # Initialize LinesFrom and LinesTo sets
    #m.LinesFrom = Set(m.Buses, initialize=lambda b: [b for b in lines_from_b if b == b])
    #m.LinesTo = Set(m.Buses, initialize=lambda b: [b for b in lines_to_b if b == b])
    
    # parameters
    m.BusFrom = Param(m.Lines, initialize=bus_l_from)
    m.BusTo = Param(m.Lines, initialize=bus_l_to)
    m.BusContaining = Param(m.Generators, initialize=bus_containing)
    m.Susceptance = Param(m.Lines, initialize=susceptance,
                          within=NonNegativeReals)
    m.ThermalLimit = Param(m.Lines, initialize=thermal_limit,
                           within=NonNegativeReals)
    m.MinimumPowerOutput = Param(m.Generators, initialize=generator_lb)
    m.MaximumPowerOutput = Param(m.Generators, initialize=generator_ub,
                                 within=NonNegativeReals)
    m.Demand = Param(m.Buses, initialize=demand,
                     mutable=True, within=NonNegativeReals)
    return


#
# Model Builder For Low Fidelity
#

def LF_builder(data,args):
    cwd = os.getcwd()
    data_file = data['data_dir']

    print('loading data')
    data_loc = json.load(open(data_file, 'r'))

    thermal_gens = data_loc['thermal_generators']
    renewable_gens = data_loc['renewable_generators']

    time_periods = {t+1 : t for t in range(data_loc['time_periods'])}

    gen_startup_categories = {g : list(range(0, len(gen['startup']))) for (g, gen) in thermal_gens.items()}
    gen_pwl_points = {g : list(range(0, len(gen['piecewise_production']))) for (g, gen) in thermal_gens.items()}

    print('building model')
    m = ConcreteModel()

    m.cg = Var(thermal_gens.keys(), time_periods.keys())
    m.pg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)  
    m.rg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)  
    m.pw = Var(renewable_gens.keys(), time_periods.keys(), within=NonNegativeReals)
    m.ug = Var(thermal_gens.keys(), time_periods.keys(), within=Binary) 
    m.vg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary) 
    m.wg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary) 

    m.dg = Var(((g,s,t) for g in thermal_gens for s in gen_startup_categories[g] for t in time_periods), within=Binary) ##
    m.lg = Var(((g,l,t) for g in thermal_gens for l in gen_pwl_points[g] for t in time_periods), within=UnitInterval) ##

    m.obj = Objective(expr=sum(
                            sum(
                                m.cg[g,t] + gen['piecewise_production'][0]['cost']*m.ug[g,t]
                                + sum( gen_startup['cost']*m.dg[g,s,t] for (s, gen_startup) in enumerate(gen['startup']))
                            for t in time_periods)
                            for g, gen in thermal_gens.items() )
                            ) #(1)

    m.demand = Constraint(time_periods.keys())
    m.reserves = Constraint(time_periods.keys())
    for t,t_idx in time_periods.items():
        m.demand[t] = sum( m.pg[g,t]+gen['power_output_minimum']*m.ug[g,t] for (g, gen) in thermal_gens.items() ) + sum( m.pw[w,t] for w in renewable_gens ) == data_loc['demand'][t_idx] #(2)
        m.reserves[t] = sum( m.rg[g,t] for g in thermal_gens ) >= data_loc['reserves'][t_idx] #(3)

    m.uptimet0 = Constraint(thermal_gens.keys())
    m.downtimet0 = Constraint(thermal_gens.keys())
    m.logicalt0 = Constraint(thermal_gens.keys())
    m.startupt0 = Constraint(thermal_gens.keys())

    m.rampupt0 = Constraint(thermal_gens.keys())
    m.rampdownt0 = Constraint(thermal_gens.keys())
    m.shutdownt0 = Constraint(thermal_gens.keys())

    for g, gen in thermal_gens.items():
        if gen['unit_on_t0'] == 1:
            if gen['time_up_minimum'] - gen['time_up_t0'] >= 1:
                m.uptimet0[g] = sum( (m.ug[g,t] - 1) for t in range(1, min(gen['time_up_minimum'] - gen['time_up_t0'], data_loc['time_periods'])+1)) == 0 #(4)
        elif gen['unit_on_t0'] == 0:
            if gen['time_down_minimum'] - gen['time_down_t0'] >= 1:
                m.downtimet0[g] = sum( m.ug[g,t] for t in range(1, min(gen['time_down_minimum'] - gen['time_down_t0'], data_loc['time_periods'])+1)) == 0 #(5)
        else:
            raise Exception('Invalid unit_on_t0 for generator {}, unit_on_t0={}'.format(g, gen['unit_on_t0']))

        m.logicalt0[g] = m.ug[g,1] - gen['unit_on_t0'] == m.vg[g,1] - m.wg[g,1] #(6)

        startup_expr = sum( 
                            sum( m.dg[g,s,t] 
                                    for t in range(
                                                    max(1,gen['startup'][s+1]['lag']-gen['time_down_t0']+1),
                                                    min(gen['startup'][s+1]['lag']-1,data_loc['time_periods'])+1
                                                )
                                ) 
                        for s,_ in enumerate(gen['startup'][:-1])) ## all but last
        if isinstance(startup_expr, int):
            pass
        else:
            m.startupt0[g] = startup_expr == 0 #(7)

        m.rampupt0[g] = m.pg[g,1] + m.rg[g,1] - gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) <= gen['ramp_up_limit'] #(8)

        m.rampdownt0[g] = gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) - m.pg[g,1] <= gen['ramp_down_limit'] #(9)


        shutdown_constr = gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) <= gen['unit_on_t0']*(gen['power_output_maximum'] - gen['power_output_minimum']) - max((gen['power_output_maximum'] - gen['ramp_shutdown_limit']),0)*m.wg[g,1] #(10)

        if isinstance(shutdown_constr, bool):
            pass
        else:
            m.shutdownt0[g] = shutdown_constr

    m.mustrun = Constraint(thermal_gens.keys(), time_periods.keys())
    m.logical = Constraint(thermal_gens.keys(), time_periods.keys())
    m.uptime = Constraint(thermal_gens.keys(), time_periods.keys())
    m.downtime = Constraint(thermal_gens.keys(), time_periods.keys())
    m.startup_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.gen_limit1 = Constraint(thermal_gens.keys(), time_periods.keys())
    m.gen_limit2 = Constraint(thermal_gens.keys(), time_periods.keys())
    m.ramp_up = Constraint(thermal_gens.keys(), time_periods.keys())
    m.ramp_down = Constraint(thermal_gens.keys(), time_periods.keys())
    m.power_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.cost_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.on_select = Constraint(thermal_gens.keys(), time_periods.keys())

    for g, gen in thermal_gens.items():
        for t in time_periods:
            m.mustrun[g,t] = m.ug[g,t] >= gen['must_run'] #(11)

            if t > 1:
                m.logical[g,t] = m.ug[g,t] - m.ug[g,t-1] == m.vg[g,t] - m.wg[g,t] #(12)

            UT = min(gen['time_up_minimum'],data_loc['time_periods'])
            if t >= UT:
                m.uptime[g,t] = sum(m.vg[g,t] for t in range(t-UT+1, t+1)) <= m.ug[g,t] #(13)
            DT = min(gen['time_down_minimum'],data_loc['time_periods'])
            if t >= DT:
                m.downtime[g,t] = sum(m.wg[g,t] for t in range(t-DT+1, t+1)) <= 1-m.ug[g,t] #(14)
            m.startup_select[g,t] = m.vg[g,t] == sum(m.dg[g,s,t] for s,_ in enumerate(gen['startup'])) #(16)

            m.gen_limit1[g,t] = m.pg[g,t]+m.rg[g,t] <= (gen['power_output_maximum'] - gen['power_output_minimum'])*m.ug[g,t] - max((gen['power_output_maximum'] - gen['ramp_startup_limit']),0)*m.vg[g,t] #(17)

            if t < len(time_periods): 
                m.gen_limit2[g,t] = m.pg[g,t]+m.rg[g,t] <= (gen['power_output_maximum'] - gen['power_output_minimum'])*m.ug[g,t] - max((gen['power_output_maximum'] - gen['ramp_shutdown_limit']),0)*m.wg[g,t+1] #(18)

            if t > 1:
                m.ramp_up[g,t] = m.pg[g,t]+m.rg[g,t] - m.pg[g,t-1] <= gen['ramp_up_limit'] #(19)
                m.ramp_down[g,t] = m.pg[g,t-1] - m.pg[g,t] <= gen['ramp_down_limit'] #(20

            piece_mw1 = gen['piecewise_production'][0]['mw']
            piece_cost1 = gen['piecewise_production'][0]['cost']
            m.power_select[g,t] = m.pg[g,t] == sum( (piece['mw'] - piece_mw1)*m.lg[g,l,t] for l,piece in enumerate(gen['piecewise_production'])) #(21)
            m.cost_select[g,t] = m.cg[g,t] == sum( (piece['cost'] - piece_cost1)*m.lg[g,l,t] for l,piece in enumerate(gen['piecewise_production'])) #(22)
            m.on_select[g,t] = m.ug[g,t] == sum(m.lg[g,l,t] for l,_ in enumerate(gen['piecewise_production'])) #(23)

    m.startup_allowed = Constraint(m.dg.index_set())
    for g, gen in thermal_gens.items():
        for s,_ in enumerate(gen['startup'][:-1]): ## all but last
            for t in time_periods:
                if t >= gen['startup'][s+1]['lag']:
                    m.startup_allowed[g,s,t] = m.dg[g,s,t] <= sum(m.wg[g,t-i] for i in range(gen['startup'][s]['lag'], gen['startup'][s+1]['lag'])) #(15)

    for w, gen in renewable_gens.items():
        for t, t_idx in time_periods.items():
            m.pw[w,t].setlb(gen['power_output_minimum'][t_idx]) #(24)
            m.pw[w,t].setub(gen['power_output_maximum'][t_idx]) #(24)
    return m
#
# Model Builder For Low Fidelity w/ Load Shed
#
def LF_builder_SHED(data,args):
    cwd = os.getcwd()
    data_file = data['data_dir']

    print('loading data')
    data_loc = json.load(open(data_file, 'r'))

    thermal_gens = data_loc['thermal_generators']
    renewable_gens = data_loc['renewable_generators']

    time_periods = {t+1 : t for t in range(data_loc['time_periods'])}

    gen_startup_categories = {g : list(range(0, len(gen['startup']))) for (g, gen) in thermal_gens.items()}
    gen_pwl_points = {g : list(range(0, len(gen['piecewise_production']))) for (g, gen) in thermal_gens.items()}

    print('building model')
    m = ConcreteModel()

    m.cg = Var(thermal_gens.keys(), time_periods.keys())
    m.pg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)  
    m.rg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)  
    m.pw = Var(renewable_gens.keys(), time_periods.keys(), within=NonNegativeReals)
    m.ug = Var(thermal_gens.keys(), time_periods.keys(), within=Binary) 
    m.vg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary) 
    m.wg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary) 

    m.dg = Var(((g,s,t) for g in thermal_gens for s in gen_startup_categories[g] for t in time_periods), within=Binary) ##
    m.lg = Var(((g,l,t) for g in thermal_gens for l in gen_pwl_points[g] for t in time_periods), within=UnitInterval) ##

   

    m.demand = Constraint(time_periods.keys())
    m.reserves = Constraint(time_periods.keys())
    m.LOAD_SHED= Var(time_periods.keys(),within=NonNegativeReals)
    for t,t_idx in time_periods.items():
        m.demand[t] = sum( m.pg[g,t]+gen['power_output_minimum']*m.ug[g,t] for (g, gen) in thermal_gens.items() ) + sum( m.pw[w,t] for w in renewable_gens ) == data_loc['demand'][t_idx]-m.LOAD_SHED[t_idx+1] #(2)
        m.reserves[t] = sum( m.rg[g,t] for g in thermal_gens ) >= data_loc['reserves'][t_idx] #(3)
    
    m.obj = Objective(expr=sum(
                            sum(
                                m.cg[g,t] + gen['piecewise_production'][0]['cost']*m.ug[g,t]
                                + sum( gen_startup['cost']*m.dg[g,s,t] for (s, gen_startup) in enumerate(gen['startup']))
                            for t in time_periods)
                            for g, gen in thermal_gens.items() )
                            +sum(1000*m.LOAD_SHED[t] for t in time_periods.keys()) )#(1)
    
    m.uptimet0 = Constraint(thermal_gens.keys())
    m.downtimet0 = Constraint(thermal_gens.keys())
    m.logicalt0 = Constraint(thermal_gens.keys())
    m.startupt0 = Constraint(thermal_gens.keys())

    m.rampupt0 = Constraint(thermal_gens.keys())
    m.rampdownt0 = Constraint(thermal_gens.keys())
    m.shutdownt0 = Constraint(thermal_gens.keys())

    for g, gen in thermal_gens.items():
        if gen['unit_on_t0'] == 1:
            if gen['time_up_minimum'] - gen['time_up_t0'] >= 1:
                m.uptimet0[g] = sum( (m.ug[g,t] - 1) for t in range(1, min(gen['time_up_minimum'] - gen['time_up_t0'], data_loc['time_periods'])+1)) == 0 #(4)
        elif gen['unit_on_t0'] == 0:
            if gen['time_down_minimum'] - gen['time_down_t0'] >= 1:
                m.downtimet0[g] = sum( m.ug[g,t] for t in range(1, min(gen['time_down_minimum'] - gen['time_down_t0'], data_loc['time_periods'])+1)) == 0 #(5)
        else:
            raise Exception('Invalid unit_on_t0 for generator {}, unit_on_t0={}'.format(g, gen['unit_on_t0']))

        m.logicalt0[g] = m.ug[g,1] - gen['unit_on_t0'] == m.vg[g,1] - m.wg[g,1] #(6)

        startup_expr = sum( 
                            sum( m.dg[g,s,t] 
                                    for t in range(
                                                    max(1,gen['startup'][s+1]['lag']-gen['time_down_t0']+1),
                                                    min(gen['startup'][s+1]['lag']-1,data_loc['time_periods'])+1
                                                )
                                ) 
                        for s,_ in enumerate(gen['startup'][:-1])) ## all but last
        if isinstance(startup_expr, int):
            pass
        else:
            m.startupt0[g] = startup_expr == 0 #(7)

        m.rampupt0[g] = m.pg[g,1] + m.rg[g,1] - gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) <= gen['ramp_up_limit'] #(8)

        m.rampdownt0[g] = gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) - m.pg[g,1] <= gen['ramp_down_limit'] #(9)


        shutdown_constr = gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) <= gen['unit_on_t0']*(gen['power_output_maximum'] - gen['power_output_minimum']) - max((gen['power_output_maximum'] - gen['ramp_shutdown_limit']),0)*m.wg[g,1] #(10)

        if isinstance(shutdown_constr, bool):
            pass
        else:
            m.shutdownt0[g] = shutdown_constr

    m.mustrun = Constraint(thermal_gens.keys(), time_periods.keys())
    m.logical = Constraint(thermal_gens.keys(), time_periods.keys())
    m.uptime = Constraint(thermal_gens.keys(), time_periods.keys())
    m.downtime = Constraint(thermal_gens.keys(), time_periods.keys())
    m.startup_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.gen_limit1 = Constraint(thermal_gens.keys(), time_periods.keys())
    m.gen_limit2 = Constraint(thermal_gens.keys(), time_periods.keys())
    m.ramp_up = Constraint(thermal_gens.keys(), time_periods.keys())
    m.ramp_down = Constraint(thermal_gens.keys(), time_periods.keys())
    m.power_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.cost_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.on_select = Constraint(thermal_gens.keys(), time_periods.keys())

    for g, gen in thermal_gens.items():
        for t in time_periods:
            m.mustrun[g,t] = m.ug[g,t] >= gen['must_run'] #(11)

            if t > 1:
                m.logical[g,t] = m.ug[g,t] - m.ug[g,t-1] == m.vg[g,t] - m.wg[g,t] #(12)

            UT = min(gen['time_up_minimum'],data_loc['time_periods'])
            if t >= UT:
                m.uptime[g,t] = sum(m.vg[g,t] for t in range(t-UT+1, t+1)) <= m.ug[g,t] #(13)
            DT = min(gen['time_down_minimum'],data_loc['time_periods'])
            if t >= DT:
                m.downtime[g,t] = sum(m.wg[g,t] for t in range(t-DT+1, t+1)) <= 1-m.ug[g,t] #(14)
            m.startup_select[g,t] = m.vg[g,t] == sum(m.dg[g,s,t] for s,_ in enumerate(gen['startup'])) #(16)

            m.gen_limit1[g,t] = m.pg[g,t]+m.rg[g,t] <= (gen['power_output_maximum'] - gen['power_output_minimum'])*m.ug[g,t] - max((gen['power_output_maximum'] - gen['ramp_startup_limit']),0)*m.vg[g,t] #(17)

            if t < len(time_periods): 
                m.gen_limit2[g,t] = m.pg[g,t]+m.rg[g,t] <= (gen['power_output_maximum'] - gen['power_output_minimum'])*m.ug[g,t] - max((gen['power_output_maximum'] - gen['ramp_shutdown_limit']),0)*m.wg[g,t+1] #(18)

            if t > 1:
                m.ramp_up[g,t] = m.pg[g,t]+m.rg[g,t] - m.pg[g,t-1] <= gen['ramp_up_limit'] #(19)
                m.ramp_down[g,t] = m.pg[g,t-1] - m.pg[g,t] <= gen['ramp_down_limit'] #(20

            piece_mw1 = gen['piecewise_production'][0]['mw']
            piece_cost1 = gen['piecewise_production'][0]['cost']
            m.power_select[g,t] = m.pg[g,t] == sum( (piece['mw'] - piece_mw1)*m.lg[g,l,t] for l,piece in enumerate(gen['piecewise_production'])) #(21)
            m.cost_select[g,t] = m.cg[g,t] == sum( (piece['cost'] - piece_cost1)*m.lg[g,l,t] for l,piece in enumerate(gen['piecewise_production'])) #(22)
            m.on_select[g,t] = m.ug[g,t] == sum(m.lg[g,l,t] for l,_ in enumerate(gen['piecewise_production'])) #(23)

    m.startup_allowed = Constraint(m.dg.index_set())
    for g, gen in thermal_gens.items():
        for s,_ in enumerate(gen['startup'][:-1]): ## all but last
            for t in time_periods:
                if t >= gen['startup'][s+1]['lag']:
                    m.startup_allowed[g,s,t] = m.dg[g,s,t] <= sum(m.wg[g,t-i] for i in range(gen['startup'][s]['lag'], gen['startup'][s+1]['lag'])) #(15)

    for w, gen in renewable_gens.items():
        for t, t_idx in time_periods.items():
            m.pw[w,t].setlb(gen['power_output_minimum'][t_idx]) #(24)
            m.pw[w,t].setub(gen['power_output_maximum'][t_idx]) #(24)
    return m
#
# Model Builder For High Fidelity
#

def HF_builder(data,args):

    cwd = os.getcwd()
    data_file = data['data_dir']

    print('loading data')
    data_loc = json.load(open(data_file, 'r'))
    bus_df=pd.read_csv('data/buses.csv')
    gen_df=pd.read_csv('data/generators.csv')
    line_df=pd.read_csv('data/lines.csv')

    thermal_gens = data_loc['thermal_generators']
    renewable_gens = data_loc['renewable_generators']

    time_periods = {t+1 : t for t in range(data_loc['time_periods'])}

    gen_startup_categories = {g : list(range(0, len(gen['startup']))) for (g, gen) in thermal_gens.items()}
    gen_pwl_points = {g : list(range(0, len(gen['piecewise_production']))) for (g, gen) in thermal_gens.items()}

    print('building model')
    m = ConcreteModel()

    m.cg = Var(thermal_gens.keys(), time_periods.keys())
    m.pg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)  
    m.rg = Var(thermal_gens.keys(), time_periods.keys(), within=NonNegativeReals)  
    m.pw = Var(renewable_gens.keys(), time_periods.keys(), within=NonNegativeReals)
    m.ug = Var(thermal_gens.keys(), time_periods.keys(), within=Binary) 
    m.vg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary) 
    m.wg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary) 

    m.dg = Var(((g,s,t) for g in thermal_gens for s in gen_startup_categories[g] for t in time_periods), within=Binary) ##
    m.lg = Var(((g,l,t) for g in thermal_gens for l in gen_pwl_points[g] for t in time_periods), within=UnitInterval) ##

    m.uptimet0 = Constraint(thermal_gens.keys())
    m.downtimet0 = Constraint(thermal_gens.keys())
    m.logicalt0 = Constraint(thermal_gens.keys())
    m.startupt0 = Constraint(thermal_gens.keys())

    m.rampupt0 = Constraint(thermal_gens.keys())
    m.rampdownt0 = Constraint(thermal_gens.keys())
    m.shutdownt0 = Constraint(thermal_gens.keys())

    for g, gen in thermal_gens.items():
        if gen['unit_on_t0'] == 1:
            if gen['time_up_minimum'] - gen['time_up_t0'] >= 1:
                m.uptimet0[g] = sum( (m.ug[g,t] - 1) for t in range(1, min(gen['time_up_minimum'] - gen['time_up_t0'], data_loc['time_periods'])+1)) == 0 #(4)
        elif gen['unit_on_t0'] == 0:
            if gen['time_down_minimum'] - gen['time_down_t0'] >= 1:
                m.downtimet0[g] = sum( m.ug[g,t] for t in range(1, min(gen['time_down_minimum'] - gen['time_down_t0'], data_loc['time_periods'])+1)) == 0 #(5)
        else:
            raise Exception('Invalid unit_on_t0 for generator {}, unit_on_t0={}'.format(g, gen['unit_on_t0']))

        m.logicalt0[g] = m.ug[g,1] - gen['unit_on_t0'] == m.vg[g,1] - m.wg[g,1] #(6)

        startup_expr = sum( 
                            sum( m.dg[g,s,t] 
                                    for t in range(
                                                    max(1,gen['startup'][s+1]['lag']-gen['time_down_t0']+1),
                                                    min(gen['startup'][s+1]['lag']-1,data_loc['time_periods'])+1
                                                )
                                ) 
                        for s,_ in enumerate(gen['startup'][:-1])) ## all but last
        if isinstance(startup_expr, int):
            pass
        else:
            m.startupt0[g] = startup_expr == 0 #(7)

        m.rampupt0[g] = m.pg[g,1] + m.rg[g,1] - gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) <= gen['ramp_up_limit'] #(8)

        m.rampdownt0[g] = gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) - m.pg[g,1] <= gen['ramp_down_limit'] #(9)


        shutdown_constr = gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) <= gen['unit_on_t0']*(gen['power_output_maximum'] - gen['power_output_minimum']) - max((gen['power_output_maximum'] - gen['ramp_shutdown_limit']),0)*m.wg[g,1] #(10)

        if isinstance(shutdown_constr, bool):
            pass
        else:
            m.shutdownt0[g] = shutdown_constr
    m.mustrun = Constraint(thermal_gens.keys(), time_periods.keys())
    m.logical = Constraint(thermal_gens.keys(), time_periods.keys())
    m.uptime = Constraint(thermal_gens.keys(), time_periods.keys())
    m.downtime = Constraint(thermal_gens.keys(), time_periods.keys())
    m.startup_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.gen_limit1 = Constraint(thermal_gens.keys(), time_periods.keys())
    m.gen_limit2 = Constraint(thermal_gens.keys(), time_periods.keys())
    m.ramp_up = Constraint(thermal_gens.keys(), time_periods.keys())
    m.ramp_down = Constraint(thermal_gens.keys(), time_periods.keys())
    m.power_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.cost_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.on_select = Constraint(thermal_gens.keys(), time_periods.keys())

    for g, gen in thermal_gens.items():
        for t in time_periods:
            m.mustrun[g,t] = m.ug[g,t] >= gen['must_run'] #(11)

            if t > 1:
                m.logical[g,t] = m.ug[g,t] - m.ug[g,t-1] == m.vg[g,t] - m.wg[g,t] #(12)

            UT = min(gen['time_up_minimum'],data_loc['time_periods'])
            if t >= UT:
                m.uptime[g,t] = sum(m.vg[g,t] for t in range(t-UT+1, t+1)) <= m.ug[g,t] #(13)
            DT = min(gen['time_down_minimum'],data_loc['time_periods'])
            if t >= DT:
                m.downtime[g,t] = sum(m.wg[g,t] for t in range(t-DT+1, t+1)) <= 1-m.ug[g,t] #(14)
            m.startup_select[g,t] = m.vg[g,t] == sum(m.dg[g,s,t] for s,_ in enumerate(gen['startup'])) #(16)

            m.gen_limit1[g,t] = m.pg[g,t]+m.rg[g,t] <= (gen['power_output_maximum'] - gen['power_output_minimum'])*m.ug[g,t] - max((gen['power_output_maximum'] - gen['ramp_startup_limit']),0)*m.vg[g,t] #(17)

            if t < len(time_periods): 
                m.gen_limit2[g,t] = m.pg[g,t]+m.rg[g,t] <= (gen['power_output_maximum'] - gen['power_output_minimum'])*m.ug[g,t] - max((gen['power_output_maximum'] - gen['ramp_shutdown_limit']),0)*m.wg[g,t+1] #(18)

            if t > 1:
                m.ramp_up[g,t] = m.pg[g,t]+m.rg[g,t] - m.pg[g,t-1] <= gen['ramp_up_limit'] #(19)
                m.ramp_down[g,t] = m.pg[g,t-1] - m.pg[g,t] <= gen['ramp_down_limit'] #(20

            piece_mw1 = gen['piecewise_production'][0]['mw']
            piece_cost1 = gen['piecewise_production'][0]['cost']
            m.power_select[g,t] = m.pg[g,t] == sum( (piece['mw'] - piece_mw1)*m.lg[g,l,t] for l,piece in enumerate(gen['piecewise_production'])) #(21)
            m.cost_select[g,t] = m.cg[g,t] == sum( (piece['cost'] - piece_cost1)*m.lg[g,l,t] for l,piece in enumerate(gen['piecewise_production'])) #(22)
            m.on_select[g,t] = m.ug[g,t] == sum(m.lg[g,l,t] for l,_ in enumerate(gen['piecewise_production'])) #(23)

    m.startup_allowed = Constraint(m.dg.index_set())
    for g, gen in thermal_gens.items():
        for s,_ in enumerate(gen['startup'][:-1]): ## all but last
            for t in time_periods:
                if t >= gen['startup'][s+1]['lag']:
                    m.startup_allowed[g,s,t] = m.dg[g,s,t] <= sum(m.wg[g,t-i] for i in range(gen['startup'][s]['lag'], gen['startup'][s+1]['lag'])) #(15)

    for w, gen in renewable_gens.items():
        for t, t_idx in time_periods.items():
            m.pw[w,t].setlb(gen['power_output_minimum'][t_idx]) #(24)
            m.pw[w,t].setub(gen['power_output_maximum'][t_idx]) #(24)

    add_data_to_pyomo(m,bus_df,line_df,gen_df)
    demand_signal=data_loc['demand']
    total_Dem=np.sum(m.Demand[b].value for b in m.Buses)
    m.nominal_demand_values = Param(m.Buses, initialize={b: m.Demand[b]/total_Dem for b in m.Buses})
    m.Demand_t = Param(m.Buses, time_periods.keys(), initialize=lambda model, bus, time: 
                                m.nominal_demand_values[bus]  * demand_signal[int(time)-1], 
                                mutable=True)
    m.load_shed = Var(m.Buses,time_periods.keys(), domain=NonNegativeReals)
    m.power_flow = Var(m.Lines,time_periods.keys(), domain=Reals)
    @m.Constraint(m.Buses,time_periods.keys())
    def balance(m, b,t):
        return sum(m.power_flow[k,t] for k in m.LinesTo[b]) - \
            sum(m.power_flow[k,t] for k in m.LinesFrom[b]) + \
            sum(m.pw[g,t] for g in m.GeneratorsAtBus[b] if g in renewable_gens.keys()) + \
            sum(m.pg[g,t] + thermal_gens[g]['power_output_minimum']*m.ug[g,t] for g in m.GeneratorsAtBus[b] if g in thermal_gens.keys()) + \
            m.load_shed[b,t] == m.Demand_t[b, t]


    m.angle = Var(m.Buses,time_periods.keys(), domain=Reals, bounds=(-pi, pi))
    for t in time_periods.keys():
        m.angle[113,t].fix(0.0)


    @m.Constraint(m.Lines,time_periods.keys())
    def ohms_law_lb(m, k,t):
        return m.power_flow[k,t] >=10*m.Susceptance[k]*(
            m.angle[m.BusFrom[k],t] - m.angle[m.BusTo[k],t])

    @m.Constraint(m.Lines,time_periods.keys())
    def ohms_law_ub(m, k,t):
        return m.power_flow[k,t] <= 10*m.Susceptance[k]*(
            m.angle[m.BusFrom[k],t] - m.angle[m.BusTo[k],t])


    @m.Constraint(m.Lines,time_periods.keys())
    def power_flow_lb(m, k,t):
        return m.power_flow[k,t] >= -m.ThermalLimit[k]

    @m.Constraint(m.Lines,time_periods.keys())
    def power_flow_ub(m, k,t):
        return m.power_flow[k,t] <= m.ThermalLimit[k]

    @m.Constraint(m.Buses, time_periods.keys())
    def load_shed_lb(m, b, t):
        return m.load_shed[b, t] >= 0 

    @m.Constraint(m.Buses, time_periods.keys())
    def load_shed_ub(m, b, t):
        return m.load_shed[b, t] <= m.Demand_t[b, t]


    m.obj = Objective(expr=sum(
                            sum(
                                m.cg[g,t] + gen['piecewise_production'][0]['cost']*m.ug[g,t]
                                + sum( gen_startup['cost']*m.dg[g,s,t] for (s, gen_startup) in enumerate(gen['startup']))
                            for t in time_periods)
                            for g, gen in thermal_gens.items() )
                            +sum(sum(1000*m.load_shed[b, t] for b in m.Buses) for t in time_periods.keys())
                            ) #(1)
    m.reserves = Constraint(time_periods.keys())
    for t,t_idx in time_periods.items():
    #    m.demand[t] = sum( m.pg[g,t]+gen['power_output_minimum']*m.ug[g,t] for (g, gen) in thermal_gens.items() ) + sum( m.pw[w,t] for w in renewable_gens ) == data['demand'][t_idx] #(2)
        m.reserves[t] = sum( m.rg[g,t] for g in thermal_gens ) >= data_loc['reserves'][t_idx] #(3)
    return m
