"""27041 Emir Kantul Solving Slant Puzzle Using CSP """

"""
    Difficulty Levels
    Easy: 5x5 Easy Slant        https://www.puzzle-slant.com/
    Normal: 5x5 Normal Slant    https://www.puzzle-slant.com/?size=1
    Hard: 7X7 Easy Slant        https://www.puzzle-slant.com/?size=2
"""
import os
import glob
import sys
import time
from ortools.sat.python import cp_model

"""
    Example puzzle and the slants
    (NxN puzzle has (N-1)x(N-1) slants)

    -1 represents the "*" in the puzzle
    -1 represents the "\" in the slants
    1 represents the "/" in the slants
    
    ex_puzzle = [
        [-1, 0, -1, -1, -1, 0],
        [-1, 3, -1, 3, -1, -1],
        [1, -1, -1, -1, 3, -1],
        [-1, 3, -1, -1, 2, 2],
        [-1, -1, 4, -1, -1, -1],
        [-1, -1, -1, -1, 1, -1],
    ]

    ex_slants = [
        [-1, 1, -1, -1, -1],
        [1, 1, 1, -1, 1],
        [1, 1, 1, -1, -1],
        [1, -1, 1, 1, 1],
        [1, 1, -1, -1, -1],
    ]


    Possible loops in the puzzle
        2X2:
            ⟋ ⟍ 
            ⟍ ⟋ 

    ----------------------------------------

        3X3:
            ⟋ ⟋ ⟍ 
            ⟋ ⟋ ⟋ 
            ⟍ ⟋ ⟋ 
    ,

            ⟋ ⟍ ⟍ 
            ⟍ ⟍ ⟍ 
            ⟍ ⟍ ⟋ 

    ----------------------------------------

        4x4:
            ⟋ ⟋ ⟍ ⟍ 
            ⟋ ⟋ ⟍ ⟍ 
            ⟍ ⟍ ⟍ ⟋ 
            ⟍ ⟍ ⟋ ⟋ 

    ----------------------------------------

        5x5:
            ⟋ ⟋ ⟍ ⟍ ⟍ 
            ⟋ ⟍ ⟍ ⟍ ⟍ 
            ⟍ ⟍ ⟋ ⟍ ⟍ 
            ⟍ ⟍ ⟍ ⟍ ⟋ 
            ⟍ ⟍ ⟍ ⟋ ⟋ 

    ,

            ⟋ ⟋ ⟋ ⟍ ⟍ 
            ⟋ ⟋ ⟋ ⟋ ⟍ 
            ⟋ ⟋ ⟍ ⟋ ⟋ 
            ⟍ ⟋ ⟋ ⟋ ⟋ 
            ⟍ ⟍ ⟋ ⟋ ⟋ 

    ----------------------------------------

        6x6:
            ⟋ ⟋ ⟋ ⟍ ⟍ ⟍
            ⟋ ⟋ ⟍ ⟍ ⟍ ⟍
            ⟋ ⟍ ⟍ ⟋ ⟍ ⟍
            ⟍ ⟍ ⟋ ⟋ ⟋ ⟋
            ⟍ ⟍ ⟋ ⟋ ⟋ ⟋
            ⟍ ⟍ ⟍ ⟋ ⟋ ⟋

    ----------------------------------------

        7x7:
            ⟋ ⟋ ⟋ ⟍ ⟍ ⟍ ⟍
            ⟋ ⟋ ⟍ ⟍ ⟍ ⟍ ⟍
            ⟋ ⟍ ⟍ ⟋ ⟍ ⟍ ⟍
            ⟍ ⟍ ⟋ ⟋ ⟋ ⟍ ⟍
            ⟍ ⟍ ⟋ ⟋ ⟍ ⟍ ⟋
            ⟍ ⟍ ⟍ ⟍ ⟍ ⟋ ⟋
            ⟍ ⟍ ⟍ ⟍ ⟋ ⟋ ⟋

    ,

            ⟋ ⟋ ⟋ ⟋ ⟍ ⟍ ⟍
            ⟋ ⟋ ⟋ ⟋ ⟋ ⟍ ⟍
            ⟋ ⟋ ⟍ ⟍ ⟋ ⟋ ⟍
            ⟋ ⟋ ⟍ ⟍ ⟍ ⟋ ⟋
            ⟍ ⟋ ⟋ ⟍ ⟋ ⟋ ⟋
            ⟍ ⟍ ⟋ ⟋ ⟋ ⟋ ⟋
            ⟍ ⟍ ⟍ ⟋ ⟋ ⟋ ⟋

"""

# This part used to find all the solutions for the given puzzle and slants durning the development time
# but since there is no optimal solution and feasible solutions are enough, this part is commented out.
""" """


class SlantPuzzleSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, puzzle, slants):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__puzzle = puzzle
        self.__slants = slants
        self.__solution_count = 0
        self.__start_time = time.time()

    """
        Since OR-Tools does not support string values
        we need to represent the solution as integers
        and convert them to string representation when 
        we need with this function. 
    """

    def int_to_str_representation(self, i, conversion):
        if conversion == "solution":
            return "\\" if self.Value(i) == -1 else "/"
        elif conversion == "puzzle":
            return "*" if self.Value(i) == -1 else str(self.Value(i))

    def solution_count(self):
        return self.__solution_count

    """
        This function is called when a solution is found.
        It prints the solution and the puzzle.
    """

    def on_solution_callback(self):
        current_time = time.time()
        print(
            "Solution %i, time = %f s"
            % (self.__solution_count, current_time - self.__start_time)
        )
        self.__solution_count += 1

        for i in range(len(self.__slants) * 2 + 1):
            for j in range(len(self.__slants) * 2 + 1):
                if i % 2 == 0:
                    if j % 2 == 0:
                        print(
                            self.int_to_str_representation(
                                self.__puzzle[i // 2][j // 2], "puzzle"
                            ),
                            end="",
                        )
                    else:
                        print(" ", end="")
                else:
                    if j % 2 == 0:
                        print(" ", end="")
                    else:
                        print(
                            self.int_to_str_representation(
                                self.__slants[i // 2][j // 2], "solution"
                            ),
                            end="",
                        )
            print()


def solve_slant_puzzle(puzzle):
    # Creates the solver.
    model = cp_model.CpModel()

    # Creates the variables.
    N = len(puzzle)
    num_constraints = 0  # Add a variable to count the number of constraints.

    # There are `(N-1)x(N-1))` number of slants, because all the cells should be filled.
    # The trick here is to use domain (1,-1) for the slants. Because this way we can
    # use abs to give them value 1 and use them in calculations easier.
    slants = [
        [
            model.NewIntVarFromDomain(cp_model.Domain.FromValues([-1, 1]), f"{i}{j}")
            for i in range(N - 1)
        ]
        for j in range(N - 1)
    ]

    num_constraints += 1  # because previous line is constraining the number of slants to be (N-1)x(N-1)

    # Check every intersection node in puzzle and add the constraint if it is not a -1 ("*").
    # Constraint is that the sum of the intersections should be equal to the value of the node.
    for i in range(N):
        for j in range(N):
            if puzzle[i][j] != -1:
                # Create temporary variables to store the counts of slants for each intersection.
                count_vars = []
                if i > 0 and j > 0:
                    count_vars.append(model.NewBoolVar(f"top_left_{i}{j}"))
                    model.Add(slants[i - 1][j - 1] == -1).OnlyEnforceIf(count_vars[-1])
                    model.Add(slants[i - 1][j - 1] != -1).OnlyEnforceIf(
                        count_vars[-1].Not()
                    )
                    num_constraints += 2
                if i > 0 and j < N - 1:
                    count_vars.append(model.NewBoolVar(f"top_right_{i}{j}"))
                    model.Add(slants[i - 1][j] == 1).OnlyEnforceIf(count_vars[-1])
                    model.Add(slants[i - 1][j] != 1).OnlyEnforceIf(count_vars[-1].Not())
                    num_constraints += 2
                if i < N - 1 and j > 0:
                    count_vars.append(model.NewBoolVar(f"bottom_left_{i}{j}"))
                    model.Add(slants[i][j - 1] == 1).OnlyEnforceIf(count_vars[-1])
                    model.Add(slants[i][j - 1] != 1).OnlyEnforceIf(count_vars[-1].Not())
                    num_constraints += 2
                if i < N - 1 and j < N - 1:
                    count_vars.append(model.NewBoolVar(f"bottom_right_{i}{j}"))
                    model.Add(slants[i][j] == -1).OnlyEnforceIf(count_vars[-1])
                    model.Add(slants[i][j] != -1).OnlyEnforceIf(count_vars[-1].Not())
                    num_constraints += 2

                # Add a constraint that the sum of count_vars should be equal to the value of the node.
                model.Add(sum(count_vars) == puzzle[i][j])

                num_constraints += 1

    # Add the constraints for possible loops (only for 2x2, 3x3, 4x4, 5x5, 6x6 and 7x7)
    def add_forbidden_assignments(indices, values):
        nonlocal num_constraints
        for i in range(N - (len(indices) // 2)):
            for j in range(N - (len(indices) // 2)):
                model.AddForbiddenAssignments(
                    [slants[i + i_idx][j + j_idx] for i_idx, j_idx in indices],
                    [values],
                )
                num_constraints += 1

    # 2X2
    add_forbidden_assignments(
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        (1, -1, -1, 1),
    )

    # 3X3
    add_forbidden_assignments(
        [(0, 1), (0, 2), (1, 2), (2, 1), (2, 1), (1, 0)],
        (1, -1, 1, 1, -1, 1),
    )

    add_forbidden_assignments(
        [(0, 0), (0, 1), (1, 2), (2, 2), (2, 1), (1, 0)],
        (1, -1, -1, 1, -1, -1),
    )

    # 4X4
    add_forbidden_assignments(
        [(1, 0), (0, 1), (0, 2), (1, 3), (2, 3), (3, 2), (3, 1), (2, 0)],
        (1, 1, -1, -1, 1, 1, -1, -1),
    )

    # 5X5
    add_forbidden_assignments(
        [
            (1, 0),
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (4, 3),
            (4, 2),
            (3, 1),
            (2, 0),
        ],
        (1, 1, -1, -1, -1, 1, 1, -1, -1, -1),
    )

    add_forbidden_assignments(
        [
            (0, 3),
            (1, 4),
            (2, 4),
            (3, 3),
            (4, 2),
            (4, 1),
            (3, 0),
            (2, 0),
            (1, 1),
            (0, 2),
        ],
        (-1, -1, 1, 1, 1, -1, -1, 1, 1, 1),
    )

    # 6X6
    add_forbidden_assignments(
        [
            (2, 0),
            (1, 1),
            (0, 2),
            (3, 5),
            (4, 4),
            (5, 3),
            (0, 3),
            (1, 4),
            (2, 5),
            (0, 3),
            (1, 4),
            (2, 5),
        ],
        (1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1),
    )

    # 7X7
    add_forbidden_assignments(
        [
            (2, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (2, 5),
            (3, 6),
            (4, 6),
            (5, 5),
            (6, 4),
            (6, 3),
            (5, 2),
            (4, 1),
            (3, 0),
        ],
        (1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1),
    )

    add_forbidden_assignments(
        [
            (3, 0),
            (2, 1),
            (1, 2),
            (0, 3),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 6),
            (4, 5),
            (5, 4),
            (6, 3),
            (6, 2),
            (5, 1),
            (4, 0),
        ],
        (1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1),
    )

    # Solve the model.
    solver = cp_model.CpSolver()

    solution_printer = SlantPuzzleSolutionPrinter(puzzle, slants)
    solver.Solve(model, solution_printer)
    print(solver.ResponseStats())

    while solver.StatusName() == "INFEASIBLE":
        solver.Solve(model)
        print(solver.ResponseStats())

    # Get the number of backtracks/choices.
    num_backtracks = solver.NumBranches()

    # Get the solution.
    solution = [
        [solver.Value(slants[i][j]) for j in range(N - 1)] for i in range(N - 1)
    ]

    num_variables = (N - 1) * (N - 1)  # The number of slants is (N-1) * (N-1).

    return solution, num_variables, num_constraints, num_backtracks


def read_puzzle(file_path):
    with open(file_path, "r") as file:
        puzzle = [[int(i) if i != "*" else -1 for i in line.strip()] for line in file]
    return puzzle


def int_to_str_representation(i, conversion):
    if conversion == "solution":
        return "\\" if i == -1 else "/"
    elif conversion == "puzzle":
        return "*" if i == -1 else str(i)


def main():
    metrics = {"easy": {}, "normal": {}, "hard": {}, "test": {}}

    for metric in metrics:
        for th in range(1, 6 if metric != "test" else 4):
            file_name = f"{metric}{th}.txt"
            puzzle = read_puzzle("./puzzles/" + file_name)
            print(f"\nSolving {metric} puzzle {th}: ", file_name)

            start_time = time.time()
            (
                solution,
                num_variables,
                num_constraints,
                num_backtracks,
            ) = solve_slant_puzzle(puzzle)
            cpu_time = time.time() - start_time

            print(f"Solution:\n")
            for i in range(len(solution) * 2 + 1):
                for j in range(len(solution) * 2 + 1):
                    if i % 2 == 0:
                        if j % 2 == 0:
                            print(
                                int_to_str_representation(
                                    puzzle[i // 2][j // 2], "puzzle"
                                ),
                                end="",
                            )
                        else:
                            print(" ", end="")
                    else:
                        if j % 2 == 0:
                            print(" ", end="")
                        else:
                            print(
                                int_to_str_representation(
                                    solution[i // 2][j // 2], "solution"
                                ),
                                end="",
                            )
                print()

            print(f"\nNumber of variables: {num_variables}")
            print(f"Number of constraints: {num_constraints}")
            print(f"Number of backtracks/choices: {num_backtracks}")
            print(f"CPU time: {cpu_time:.5f} seconds\n")
            print("-" * 40)

            metrics[metric][th] = {
                "puzzle": file_name,
                "num_variables": num_variables,
                "num_constraints": num_constraints,
                "num_backtracks": num_backtracks,
                "cpu_time": cpu_time,
            }

    print("\nSummary:")
    print("Puzzle \t\t Variables \t\t Constraints \t\t Backtracks/Choices \t CPU Time")
    common_summary = {
        "num_variables": 0,
        "num_constraints": 0,
        "num_backtracks": 0,
        "cpu_time": 0,
    }
    averages = {
        "easy": common_summary,
        "normal": common_summary,
        "hard": common_summary,
        "test": common_summary,
    }
    for m in metrics:
        for i in metrics[m]:
            solution = metrics[m][i]
            print(
                f"{solution['puzzle']} \t {solution['num_variables']} \t\t\t {solution['num_constraints']} \t\t\t {solution['num_backtracks']} \t\t\t {solution['cpu_time']:.5f}"
            )
            averages[m]["num_variables"] += solution["num_variables"]
            averages[m]["num_constraints"] += solution["num_constraints"]
            averages[m]["num_backtracks"] += solution["num_backtracks"]
            averages[m]["cpu_time"] += solution["cpu_time"]

        averages[m]["num_variables"] /= 5
        averages[m]["num_constraints"] /= 5
        averages[m]["num_backtracks"] /= 5
        averages[m]["cpu_time"] /= 5

    print("\nAverages:")
    print("Metric \t\t Variables \t\t Constraints \t\t Backtracks/Choices \t CPU Time")
    for m in averages:
        print(
            f"{m} \t\t {averages[m]['num_variables']} \t\t\t {averages[m]['num_constraints']} \t\t\t {averages[m]['num_backtracks']} \t\t\t {averages[m]['cpu_time']:.5f}"
        )


if __name__ == "__main__":
    main()
