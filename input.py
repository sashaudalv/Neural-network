#!/usr/bin/python

from tkinter import *

import numpy


class MatrixInput(object):
    def __init__(self):
        self.canvas_width = 480
        self.canvas_height = 480
        self.matrix_width = 8
        self.matrix_height = 8
        self.cell_width = self.canvas_width / self.matrix_width
        self.cell_height = self.canvas_height / self.matrix_height

        self.callback = lambda x: None

        self.root = Tk()
        self.root.title("Prediction — None")
        self.root.resizable(0, 0)
        self.canvas = Canvas(self.root, bg="white", width=self.canvas_width, height=self.canvas_height)
        self.canvas.configure(cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.point)
        self.root.bind("<space>", self.clear)

        self.matrix = numpy.zeros((self.matrix_width, self.matrix_height))

    def point(self, event):
        x = int((event.x / self.canvas_width) * self.matrix_width)
        y = int((event.y / self.canvas_height) * self.matrix_height)
        self.canvas.create_rectangle(
            x * self.cell_width,
            y * self.cell_height,
            x * self.cell_width + self.cell_width,
            y * self.cell_height + self.cell_height,
            fill="black"
        )
        self.matrix[y][x] = 15
        prediction = self.callback(self.matrix.flatten())
        self.root.title("Prediction — {prediction}".format(prediction=prediction))

    def clear(self, event):
        self.matrix = numpy.zeros((self.matrix_width, self.matrix_height))
        self.canvas.delete("all")

    def show(self, callback):
        self.callback = callback
        self.root.mainloop()
