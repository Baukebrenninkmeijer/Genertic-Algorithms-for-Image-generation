from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt
from copy import deepcopy
import cv2
import numpy as np

COLOR_MODE = 'RGBA'

class Polygon:
    def __init__(self, environment, n_points=3):
        self.environment = environment
        self.width, self.height = environment.size
        self.coordinates = [(random.randint(0, self.width), random.randint(0, self.height)) for _ in range(n_points)]
        self.color = [random.randint(0, 255) for _ in range(4)]
        self.color_tuple = tuple(self.color)
        self.color_string = f'rgba{tuple(self.color)}'

    def draw(self, im: ImageDraw):
        im.polygon(self.coordinates, self.color_tuple, self.color_tuple)

    def show(self):
        # im = Image.new(COLOR_MODE, self.environment.size, (255, 255, 255, 255))
        poly_canvas = Image.new('RGBA', self.environment.size)

        draw = ImageDraw.Draw(poly_canvas, mode=COLOR_MODE)
        self.draw(draw)
        # im.paste(poly_canvas, mask=poly_canvas)
        plt.imshow(poly_canvas)

        # im.show()

    def mutate(self):
        for i, gene in enumerate(self.coordinates):
            x = gene[0] + (random.randint(0, self.width) // 10 - (self.width // 20))
            x = np.clip(x, 0, self.width)
            y = gene[1] + (random.randint(0, self.height) // 10 - (self.height // 20))
            y = np.clip(y, 0, self.height)
            self.coordinates[i] = (x, y)
        self.color = [np.clip(i+(random.randint(0, 20) - 10), 0, 255) for i in self.color]

    def calc_size(self):
        poly_canvas = Image.new('RGBA', self.environment.size)
        draw = ImageDraw.Draw(poly_canvas, mode=COLOR_MODE)
        draw.polygon(self.coordinates, (255, 255, 255, 255), (255, 255, 255, 255))
        self.size = np.sum(poly_canvas)
        return self.size


class Organism:
    def __init__(self, n_polygons, environment, mutate_prob=0.2, n_points = 4):
        self.genes = [Polygon(environment, n_points) for _ in range(n_polygons)]
        self.environment = environment
        self.mutate_prob = mutate_prob
        self.fitness = -1

    def mutate(self):
        for polygon in self.genes:
            if self.mutate_prob > random.random():
                polygon.mutate()

    def mate(self, partner):
        middle = len(self.genes) // 2
        child_a = deepcopy(self)
        child_b = deepcopy(partner)
        child_a.genes = self.genes[:middle] + partner.genes[middle:]
        child_b.genes = self.genes[middle:] + partner.genes[:middle]
        return [child_a, child_b]

    def calc_fitness(self, target):
        # canvas = deepcopy(self.environment)
        # poly = Image.new(COLOR_MODE, self.environment.size)
        #
        # draw = ImageDraw.Draw(poly, mode=COLOR_MODE)
        # for poly in self.genes:
        #     poly.draw(draw)
        # canvas.paste(poly, mask=poly)
        canvas = self.draw()
        self.fitness = mse(canvas.astype(np.float32), np.asarray(target).astype(np.float32))
        return self.fitness

    def draw(self, show=False):
        canvas = deepcopy(self.environment)
        poly_canvas = Image.new(COLOR_MODE, self.environment.size)
        sorted_genes = sorted(self.genes, key=lambda x: x.calc_size(), reverse=True)
        draw = ImageDraw.Draw(poly_canvas, mode=COLOR_MODE)
        for poly in sorted_genes:
            poly.draw(draw)
        # canvas.paste(poly_canvas, mask=poly_canvas)

        if show:
            poly_canvas.show()
        else:
            return np.asarray(poly_canvas)

        # canvas = np.ones(np.asarray(self.environment).shape)*255
        #
        # for polygon in self.genes:
        #     pts = np.array(polygon.coordinates)
        #     # pts = pts.reshape((-1, 1, 2))
        #     col = tuple(map(int, polygon.color[:-1]))
        #
        #     overlay = canvas
        #     output = canvas
        #
        #     alpha = polygon.color[-1]
        #     cv2.fillPoly(overlay, [pts], col)
        #
        #     cv2.addWeighted(overlay, 0.5, output, 1 - alpha, 0, canvas)
        # return canvas


class Population:
    def __init__(self, target, pop_size, environment, fitness_func, mutate_prob=0.2, crossover_top=None, n_poly=10):
        self.organisms = [Organism(n_poly, environment, mutate_prob=mutate_prob) for _ in range(pop_size)]
        self.environment = environment
        self.mutate_prob = mutate_prob
        self.fitness_func = fitness_func
        if crossover_top is None:
            self.crossover_top = pop_size // 5
        else:
            self.crossover_top = crossover_top
        self.target = target
        self.pop_size = pop_size
        self.best = None
        self.calc_fitness()

    def calc_fitness(self):
        for organism in self.organisms:
            organism.calc_fitness(self.target)
        self.organisms = sorted(self.organisms, key=lambda x: x.fitness)

    def mutate(self):
        for organism in self.organisms[1:]:
            organism.mutate()

    def step(self):
        # print(f'Pop size: {len(self.organisms)}')
        for i in range(self.crossover_top):
            self.organisms += self.organisms[i].mate(self.organisms[i+1])
        self.mutate()
        self.calc_fitness()
        self.organisms = self.organisms[:self.pop_size]
        if self.best is None:
            self.best = self.organisms[0]
        if self.best.fitness > self.organisms[0].fitness:
            self.best = deepcopy(self.organisms[0])
        # print([f'{x.fitness}' for x in self.organisms])


def mse(image, candidate):
    return np.mean(np.square(image - candidate))
