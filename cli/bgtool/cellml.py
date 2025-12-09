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

#===============================================================================

def valid_cellml(cellml_file: str) -> bool:
#==========================================
    file = loc.File(str(cellml_file), True)
    no_issues = True
    if file.has_issues:
        for issue in file.issues:
            print(issue.description)
        print(f'{file.issue_count} CellML validation issues...')
        no_issues = False
    else:
        simulation = loc.SedDocument(file)
        if simulation.has_issues:
            for issue in simulation.issues:
                print(issue.description)
            print(f'{simulation.issue_count} issues creating simulation from CellML...')
            no_issues = False
        else:
            simulation.simulations[0].output_end_time = 0.1
            simulation.simulations[0].number_of_steps = 10

            instance = simulation.instantiate(True)
            instance.run()
            if instance.has_issues:
                for issue in instance.issues:
                    print(issue.description)
                print(f'{instance.issue_count} issues running simulation created from CellML...')
                no_issues = False
    if no_issues:
        print('CellML is valid')
    return no_issues

#===============================================================================

if __name__ == '__main__':
    import sys
    validate_cellml(sys.argv[1])

#===============================================================================
#===============================================================================
