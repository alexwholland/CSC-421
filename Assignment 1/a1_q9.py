def isComplete(assignment):
    return None not in (assignment.values())

def select_unassigned_variable(variables, assignment):
    for var in variables:
        if assignment[var] is None:
            return var

def is_consistent(assignment, constraints):
    for constraint_violated in constraints:
        if constraint_violated(assignment):
          return False
    return True

def init_assignment(csp):
    assignment = {}
    for var in csp["VARIABLES"]:
        assignment[var] = None
    return assignment

def add_constraint(csp, constraint): 
    csp['CONSTRAINTS'].append(constraint)
    
def recursive_backtracking(assignment, csp):
    if isComplete(assignment):
        return assignment
    var = select_unassigned_variable(csp["VARIABLES"], assignment)
    for value in csp["DOMAINS"]:
        assignment[var] = value
        if is_consistent(assignment, csp["CONSTRAINTS"]):
            result = recursive_backtracking(assignment, csp)
            if result != "FAILURE":
                return result
        assignment[var] = None
    return "FAILURE"


def binary_constraint(var_pair, violations):
    (v1,v2) = var_pair
    return lambda asmt: (asmt[v1], asmt[v2]) in violations
  
# add your code for CSP-based type inference as described in the notebook 
# below. The answer to the problem provided should be named result and 
# be a dictionary with a complete assignment of the variables to types 
# as returned by the CSP backtracking method. 

def unary_constraint(variable, violations):
    return lambda asmt: asmt[variable] in violations

def generate_violations(allowed):
    return [(type1, type2, type3) for type1 in allowed
            for type2 in allowed
            for type3 in allowed
            if (type1, type2, type3) != tuple(allowed)]

def ternary_constraint(var_triple, allowed):
    (v1, v2, v3) = var_triple
    violations = generate_violations(allowed)

    return lambda asmt: (asmt[v1], asmt[v2], asmt[v3]) in violations

csp2 = {"VARIABLES": ["I", "F", "X", "Y", "Z", "W"],
        "DOMAINS": ["int", "float"],
        "CONSTRAINTS": []}

# Add unary constraints to enforce that WA and T are blue
add_constraint(csp2, unary_constraint("I", ["float"]))
add_constraint(csp2, unary_constraint("F", ["int"]))

# X = I
add_constraint(csp2, binary_constraint(("X", "I"), {("float", "float"), ("float", "int"), ("int", "float")}))
# Y = X + F
add_constraint(csp2, ternary_constraint(("Y", "X", "F"), ["float", "int", "float"]))
# Z = X + Y
add_constraint(csp2, ternary_constraint(("Z", "X", "Y"), ["float", "int", "float"]))
# W = X + I
add_constraint(csp2, ternary_constraint(("W", "X", "I"), ["int", "int", "int"]))   

result = recursive_backtracking(init_assignment(csp2), csp2)
    