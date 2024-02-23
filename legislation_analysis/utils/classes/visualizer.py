import random

import matplotlib.pyplot as plt


class Visualizer:
    """
    Visualizes textual data using matplotlib.

    parameters:
        data (df): Data to visualize.
    """

    def __init__(self):
        self.visualization_types = {
            "scatter": plt.scatter,
            "bar": plt.bar,
            "line": plt.plot,
            "heatmap": self.heatmap,
        }
        self.plts = {}

    def generate_random_color(self) -> tuple:
        """
        Generate a random RGB color.

        returns:
            (tuple) Random RGB color.
        """
        r = random.uniform(0, 1)
        g = random.uniform(0, 1)
        b = random.uniform(0, 1)
        return (r, g, b)

    def generate_random_colors(self, N) -> list:
        """
        Generate a list of random RGB colors.

        parameters:
            N (int): Number of colors to generate.

        returns:
            (list) List of random RGB colors.
        """
        colors = []
        for _ in range(N):
            color = self.generate_random_color()
            colors.append(color)
        return colors

    def heatmap(
        self, x: str, y: str, title: str, x_label: str, y_label: str
    ) -> None:
        """
        Creates a heatmap.

        parameters:
            x (str): X-axis data.
            y (str): Y-axis data.
            title (str): Title of the plot.
            x_label (str): X-axis label.
            y_label (str): Y-axis label.
        """
        print("Heatmap not implemented yet.")
        pass

    def visualize(
        self,
        _type: str,
        x: list,
        y: list,
        title: str,
        x_label: str,
        y_label: str,
    ) -> None:
        """
        Visualizes the data.

        parameters:
            type (str): Type of the plot.
            x (list): X-axis data.
            y (list): Y-axis data.
            title (str): Title of the plot.
            x_label (str): X-axis label.
            y_label (str): Y-axis label.
        """
        if not self.visualization_types.get(_type):
            raise ValueError(
                f"""
            Invalid plot type: No such plot type {_type}.
            Available plot types: {list(self.visualization_types.keys())}
            """
            )

        if _type != "heatmap":
            self.visualization_types[_type](x, y)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            self.plts[title] = plt
        else:
            self.visualization_types[_type](x, y, title, x_label, y_label)
