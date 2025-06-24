#===============================================================================
#
#  CellDL and bondgraph tools
#
#  Copyright (c) 2020 - 2025 David Brooks
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#===============================================================================

import libopencor as loc
import pytest

#===============================================================================

from bg2cellml.bondgraph import BondgraphModelSource
from bg2cellml.cellml import CellMLModel

#===============================================================================

# For pytest.approx

REL_TOLERANCE = 2e-4
ABS_TOLERANCE = 1e-6

#===============================================================================

def string_to_list(string: str) -> list[int]:
#============================================
    return [ord(x) for x in string]

def cellml_virtual_file(cellml: str) -> loc.File:
#================================================
    file = loc.File('virtual.cellml', False)
    file.contents = string_to_list(cellml)
    return file

def run_simulation(simulation):
#==============================
    if simulation.has_issues:
        raise ValueError(f'Simulation has issues: {simulation.issues[0].description}')
    instance = simulation.instantiate(True)
    instance.run()
    if instance.has_issues:
        raise ValueError(f'Simulation has issues: {instance.issues[0].description}')
    return instance.tasks[0]

def assert_equal_states_and_rates(instance_task, reference_task):
#================================================================
    assert instance_task.state_count == reference_task.state_count
    assert instance_task.rate_count == reference_task.rate_count
    for i in range(reference_task.state_count):
        assert instance_task.state(i) == pytest.approx(reference_task.state(i), rel=REL_TOLERANCE, abs=ABS_TOLERANCE)
    for i in range(reference_task.rate_count):
        assert instance_task.rate(i) == pytest.approx(reference_task.rate(i), rel=REL_TOLERANCE, abs=ABS_TOLERANCE)

def compare_simulation(bondgraph_source: str, sedml_source: str):
#===============================================================
    sedml = loc.SedDocument(loc.File(f'{sedml_source}.sedml'))
    output_end_time = sedml.simulations[0].output_end_time
    number_of_steps = sedml.simulations[0].number_of_steps

    reference = loc.SedDocument(loc.File(f'{sedml_source}.cellml'))
    reference.simulations[0].output_end_time = output_end_time
    reference.simulations[0].number_of_steps = number_of_steps
    ref_task = run_simulation(reference)

    model = BondgraphModelSource(bondgraph_source).models[0]
    cellml = CellMLModel(model).to_xml()
    simulation = loc.SedDocument(cellml_virtual_file(cellml))
    simulation.simulations[0].output_end_time = output_end_time
    simulation.simulations[0].number_of_steps = number_of_steps
    sim_task = run_simulation(simulation)

    assert_equal_states_and_rates(ref_task, sim_task)

def compare_bondgraph_models(bondgraph_source_1: str, bondgraph_source_2: str, sedml_source: str):
#=================================================================================================
    sedml = loc.SedDocument(loc.File(f'{sedml_source}.sedml'))
    output_end_time = sedml.simulations[0].output_end_time
    number_of_steps = sedml.simulations[0].number_of_steps

    reference = loc.SedDocument(loc.File(f'{sedml_source}.cellml'))
    reference.simulations[0].output_end_time = output_end_time
    reference.simulations[0].number_of_steps = number_of_steps

    model_1 = BondgraphModelSource(bondgraph_source_1).models[0]
    cellml_1 = CellMLModel(model_1).to_xml()
    simulation_1 = loc.SedDocument(cellml_virtual_file(cellml_1))
    simulation_1.simulations[0].output_end_time = output_end_time
    simulation_1.simulations[0].number_of_steps = number_of_steps
    sim_1_task = run_simulation(simulation_1)

    model_2 = BondgraphModelSource(bondgraph_source_2).models[0]
    cellml_2 = CellMLModel(model_2).to_xml()
    simulation_2 = loc.SedDocument(cellml_virtual_file(cellml_2))
    simulation_2.simulations[0].output_end_time = output_end_time
    simulation_2.simulations[0].number_of_steps = number_of_steps
    sim_2_task = run_simulation(simulation_2)

    assert_equal_states_and_rates(sim_1_task, sim_2_task)

#===============================================================================

def test_simple_electrical_circuit():
#====================================
    compare_simulation('../examples/example_A1.ttl',
                       '../pmr/Simple_electrical_circuit/FAIRDO BG example 3.1')

def test_simple_reaction():
#==========================
    compare_simulation('../examples/example_B1.ttl',
                       '../pmr/Simple_biochemical_reaction/FAIRDO BG example 3.4')

def test_simplified_reaction():
#==============================
    compare_simulation('../examples/example_B1_simplified.ttl',
                       '../pmr/Simple_biochemical_reaction/FAIRDO BG example 3.4')

def test_bondgraph_simplification():
#===================================
    compare_bondgraph_models('../examples/example_B1.ttl',
                             '../examples/example_B1_simplified.ttl',
                             '../pmr/Simple_biochemical_reaction/FAIRDO BG example 3.4')

#===============================================================================
#===============================================================================
