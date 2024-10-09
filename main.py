import numpy as np
import plotly.graph_objects as go
import random


class TSPProblem:
    def __init__(self, num_cities=5, min_distance=10, max_distance=100):
        self.num_cities = num_cities
        self.start_city = random.randint(0, num_cities - 1)
        self.distances = self._generate_distances(min_distance, max_distance)

    def _generate_distances(self, min_distance, max_distance):
        distances = {}
        for i in range(self.num_cities):
            distances[i] = [0] * self.num_cities

        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                distance = random.randint(min_distance, max_distance)
                distances[i][j] = distance
                distances[j][i] = distance

        return distances

    def goal_state(self, state):
        return (
            len(state["visited"]) == self.num_cities
            and state["current_city"] == self.start_city
        )


class TSPVisualizer:
    def __init__(self, tsp_problem):
        self.problem = tsp_problem
        self.positions = self._generate_circle_positions()

    def _generate_circle_positions(self):
        positions = {}
        for i in range(self.problem.num_cities):
            angle = 2 * np.pi * i / self.problem.num_cities
            radius = 1
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions[i] = (x, y)
        return positions

    def create_visualization(self, solution=None):
        fig = go.Figure()

        for i in range(self.problem.num_cities):
            for j in range(i + 1, self.problem.num_cities):
                start_pos = self.positions[i]
                end_pos = self.positions[j]

                fig.add_trace(
                    go.Scatter(
                        x=[start_pos[0], end_pos[0]],
                        y=[start_pos[1], end_pos[1]],
                        mode="lines",
                        line=dict(color="lightgray", width=1),
                        hoverinfo="text",
                        text=f"Distância: {self.problem.distances[i][j]}",
                        showlegend=False,
                    )
                )

                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2
                fig.add_annotation(
                    x=mid_x,
                    y=mid_y,
                    text=str(self.problem.distances[i][j]),
                    showarrow=False,
                    font=dict(size=10, color="red"),
                )

        if solution:
            path = solution["path"]
            for edge in path:
                start_pos = self.positions[edge[0]]
                end_pos = self.positions[edge[1]]

                fig.add_trace(
                    go.Scatter(
                        x=[start_pos[0], end_pos[0]],
                        y=[start_pos[1], end_pos[1]],
                        mode="lines",
                        line=dict(color="blue", width=3),
                        hoverinfo="text",
                        showlegend=False,
                    )
                )

        x_coords = [pos[0] for pos in self.positions.values()]
        y_coords = [pos[1] for pos in self.positions.values()]

        colors = ["lightblue"] * self.problem.num_cities
        colors[self.problem.start_city] = "lightgreen"

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers+text",
                marker=dict(size=30, color=colors),
                text=list(range(self.problem.num_cities)),
                textposition="middle center",
                hoverinfo="text",
                hovertext=[
                    f'Cidade {i}{" (Inicial)" if i == self.problem.start_city else ""}'
                    for i in range(self.problem.num_cities)
                ],
                showlegend=False,
            )
        )

        title_text = (
            "Solução do Problema do Caixeiro Viajante<br>"
            + f"Cidade inicial: {self.problem.start_city}<br>"
        )
        if solution:
            title_text += (
                f'Caminho: {" → ".join(map(str, [edge[0] for edge in path] + [path[-1][1]]))}<br>'
                + f'Distância Total: {solution["cost"]}'
            )

        fig.update_layout(
            title={
                "text": title_text,
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode="closest",
            plot_bgcolor="white",
            width=800,
            height=800,
            xaxis_range=[-1.5, 1.5],
            yaxis_range=[-1.5, 1.5],
            yaxis_scaleanchor="x",
        )

        return fig


class Graph:
    def __init__(self, edges, tsp_problem):
        self.vertices = edges
        self.problem = tsp_problem
        self.graph = self.mount_graph(edges)

    def mount_graph(self, edges):
        graph_edges = []
        for i in self.vertices:
            for j in self.vertices:
                if j > i:
                    graph_edges.append([i, j, self.problem.distances[i][j]])
        return sorted(graph_edges, key=lambda x: x[2])

    def find_root_edge(self, origin, i):
        if origin[i] != i:
            origin[i] = self.find_root_edge(origin, origin[i])
        return origin[i]

    def define_root_edge(self, origin, rank, x, y):
        if rank[x] < rank[y]:
            origin[x] = y
        elif rank[x] > rank[y]:
            origin[y] = x
        else:
            origin[y] = x
            rank[x] += 1

    def mst(self):
        mst = []
        origin = {i: i for i in self.vertices}
        rank = {i: 0 for i in self.vertices}
        for edge in self.graph:
            if len(mst) == len(self.vertices) - 1:
                break
            x = self.find_root_edge(origin, edge[0])
            y = self.find_root_edge(origin, edge[1])
            if x != y:
                mst.append(edge)
                self.define_root_edge(origin, rank, x, y)
        return sum([edge[2] for edge in mst])


def solve_tsp(tsp_problem):
    unvisited = set(range(tsp_problem.num_cities))
    start_city = tsp_problem.start_city
    unvisited.remove(start_city)
    graph = Graph(unvisited, tsp_problem)

    initial_state = {
        "visited": [],
        "current_city": start_city,
        "heuristic": graph.mst(),
        "cost": 0,
        "path": [],
    }
    initial_state["f"] = initial_state["cost"] + initial_state["heuristic"]
    nodes = [initial_state]

    while nodes:
        lowest_f_city = sorted(nodes, key=lambda x: (x["f"], x["cost"]))[0]
        unvisited_cities = [
            city
            for city in range(tsp_problem.num_cities)
            if city not in lowest_f_city["visited"]
        ]

        for unvisited_city in unvisited_cities:
            unvisited = set(unvisited_cities)
            unvisited.remove(unvisited_city)
            graph = Graph(unvisited, tsp_problem)

            path = lowest_f_city["path"].copy()
            path.append([lowest_f_city["current_city"], unvisited_city])

            node = {
                "visited": lowest_f_city["visited"] + [unvisited_city],
                "current_city": unvisited_city,
                "heuristic": graph.mst(),
                "cost": lowest_f_city["cost"]
                + tsp_problem.distances[lowest_f_city["current_city"]][unvisited_city],
                "path": path,
            }
            node["f"] = node["cost"] + node["heuristic"]
            nodes.append(node)

            if tsp_problem.goal_state(node):
                print(start_city)
                print(node)
                final_path = node["path"].copy()
                print(final_path)
                final_node = {
                    "visited": node["visited"] + [start_city],
                    "current_city": start_city,
                    "heuristic": 0,
                    "cost": node["cost"]
                    + tsp_problem.distances[node["current_city"]][start_city],
                    "path": final_path,
                }
                final_node["f"] = final_node["cost"]
                return final_node

        nodes.remove(lowest_f_city)

    return None


if __name__ == "__main__":
    tsp_problem = TSPProblem(num_cities=6, min_distance=10, max_distance=100)

    visualizer = TSPVisualizer(tsp_problem)

    solution = solve_tsp(tsp_problem)

    fig = visualizer.create_visualization(solution)
    fig.show()

    print(f"\nCidade inicial: {tsp_problem.start_city}")
    print(f"Matriz distâncias:")
    for i in range(tsp_problem.num_cities):
        print(tsp_problem.distances[i])
    print(
        f"\nSolução: {' → '.join(map(str, [edge[0] for edge in solution['path']] + [solution['path'][-1][1]]))}"
    )
    print(f"Distância: {solution['cost']}")
