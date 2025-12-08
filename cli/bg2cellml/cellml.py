import libopencor as loc

#===============================================================================

def validate_cellml(cellml_file: str) -> bool:
#=============================================
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
