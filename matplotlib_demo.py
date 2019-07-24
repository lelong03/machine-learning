import matplotlib.pyplot as plt
import numpy as np


def draw_line_demo():
    x = [3, 4]
    y = [5, 9]
    plt.plot(x, y)
    plt.show()


def draw_matrix_demo():
    image = np.random.rand(60, 60)
    plt.imshow(image)
    plt.show()


draw_matrix_demo()

