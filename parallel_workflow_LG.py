# "Cricket Batsman Performance Calculator"

from langgraph.graph import StateGraph, START, END
from typing import TypedDict



class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int

    sr: float
    bpb: float
    boundary_percent: float
    summary: str



def calculate_sr(state: BatsmanState):
    sr = (state['runs']/state['balls'])*100
    return {'sr': sr}


def calculate_bpb(state: BatsmanState):
    bpb = state['balls']/(state['fours'] + state['sixes'])
    return {'bpb': bpb}


def calculate_boundary_percent(state: BatsmanState):
    boundary_percent = (((state['fours'] * 4) + (state['sixes'] * 6))/state['runs'])*100
    return {'boundary_percent': boundary_percent}


def summary(state: BatsmanState):
    summary = f"""
Strike Rate - {state['sr']} \n
Balls per boundary - {state['bpb']} \n
Boundary percent - {state['boundary_percent']}
"""
    return {'summary': summary}



graph = StateGraph(BatsmanState)
# add nodes
graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percent', calculate_boundary_percent)
graph.add_node('summary', summary)
# add edges
graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percent')

graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percent', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()



print("Enter the Batsman stats:")
runs = int(input("  Runs: "))
balls = int(input("  Balls: "))
fours = int(input("  Fours: "))
sixes = int(input("  Sixes: "))

initial_state = {'runs': runs, 'balls': balls, 'fours': fours, 'sixes': sixes}
result = workflow.invoke(initial_state)

print("\n" + result['summary'])\



# initial_state = {
#     'runs': 68,
#     'balls': 34,
#     'fours': 8,
#     'sixes': 1
# }
# result = workflow.invoke(initial_state)

# print(result['summary'])


