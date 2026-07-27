"""Print the currently implemented components in the cpscheduler package."""

from prettytable import PrettyTable, TableStyle


def print_alpha() -> None:
    """Print the currently implemented scheduling environment setups."""
    from cpscheduler.environment.setups import setups

    table = PrettyTable()
    table.field_names = ["Class", "Notation"]
    table.set_style(TableStyle.MARKDOWN)

    for setup in setups.values():
        table.add_row([setup.__name__, setup.get_general_entry()])

    table.align = "c"
    print(table)


def print_beta() -> None:
    """Print the currently implemented scheduling environment constraints."""
    from cpscheduler.environment.constraints import (
        PassiveConstraint,
        constraints,
    )

    table = PrettyTable()
    table.field_names = ["Class", "Notation", "Type"]
    table.set_style(TableStyle.MARKDOWN)

    for constraint in constraints.values():
        entry = constraint.get_general_entry()
        if not entry:
            entry = "—"

        t = "Passive" if issubclass(constraint, PassiveConstraint) else "Active"

        table.add_row([constraint.__name__, entry, t])

    table.align = "c"
    print(table)


def print_gamma() -> None:
    """Print the currently implemented scheduling environment objectives."""
    from cpscheduler.environment.objectives import objectives

    table = PrettyTable()
    table.field_names = ["Class", "Notation"]
    table.set_style(TableStyle.MARKDOWN)

    for objective in objectives.values():
        entry = objective.get_general_entry()

        if not entry:
            continue

        table.add_row([objective.__name__, objective.get_general_entry()])

    table.align = "c"
    print(table)


if __name__ == "__main__":
    print_alpha()
    print()
    print_beta()
    print()
    print_gamma()
