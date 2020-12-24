from kivy.app import App
from kivy.config import Config
Config.set('graphics', 'width', 450)
Config.set('graphics', 'height', 900)
from kivy.uix.widget import Widget
from kivy.graphics import Line, Rectangle, Color
from kivy.uix.label import Label
from kivy.core.window import Window

from tensorflow import keras
import numpy as np
import math

model = keras.models.load_model('ShapeDetectionModel')

resolution = 56
step = 450 / resolution


class Touch(Widget):
    def __init__(self, **kwargs):
        super(Touch, self).__init__(**kwargs)

        self.x_size = Window.size[0]
        self.y_size = Window.size[1] / 2

        self.pixels_binary = np.zeros((resolution, resolution))

        with self.canvas:
            Color(0.3, 0.3, 0.3, 1, mode='rgba')
            Line(points=(0, Window.height / 2, Window.width, Window.height / 2))

        self.lbl_prediction = Label(size=(Window.width, Window.height / 2), text='Draw a shape', font_size=60,
                                    markup=True)
        self.add_widget(self.lbl_prediction)

    def on_touch_down(self, touch):
        x_value = math.floor(touch.pos[0] / step)
        y_value = math.floor((touch.pos[1] - 450) / step)

        if Window.size[1] / 2 < touch.pos[1] < Window.size[1]:
            with self.canvas:
                Color(1, 1, 1, 1, mode='rgba')
                self.rect = Rectangle(pos=(step * x_value, (step * y_value) + 450),
                                      size=(self.x_size / resolution, self.y_size / resolution))

                self.pixels_binary[x_value][y_value] = 1.0

                drawNeighbors(self, x_value, y_value)

        elif Window.size[1] / 2 > touch.pos[1] >= 0:
            with self.canvas:
                Color(0, 0, 0, 1, mode='rgba')

                self.rect = Rectangle(pos=(0, 450), size=(450, 450))

            self.pixels_binary = np.zeros((resolution, resolution))

            self.lbl_prediction.text = ''

    def on_touch_move(self, touch):
        x_value = math.floor(touch.pos[0] / step)
        y_value = math.floor((touch.pos[1] - 450) / step)

        if Window.size[1] / 2 < touch.pos[1] < Window.size[1]:
            with self.canvas:
                Color(1, 1, 1, 1, mode='rgba')
                self.rect = Rectangle(pos=(step * x_value, (step * y_value) + 450),
                                      size=(self.x_size / resolution, self.y_size / resolution))

                self.pixels_binary[x_value][y_value] = 1.0

                drawNeighbors(self, x_value, y_value)

    def on_touch_up(self, touch):
        if touch.pos[1] > Window.size[1] / 2:
            self.pixels_binary = np.rot90(self.pixels_binary, k=3, axes=(1, 0))

            class_names = ["Circle", "Square", "Triangle"]

            prediction = model.predict(self.pixels_binary.reshape(-1, 56, 56, 1))

            self.lbl_prediction.text = f'[color=ffffff]{class_names[np.argmax(prediction)]}[/color]'

            self.pixels_binary = np.rot90(self.pixels_binary, k=1, axes=(1, 0))


def drawNeighbors(self, x_value, y_value):
    # Left
    if x_value > 0:
        self.rect = Rectangle(pos=(step * x_value - step, (step * y_value) + 450),
                              size=(self.x_size / resolution, self.y_size / resolution))
        self.pixels_binary[x_value - 1][y_value] = 1.0

    # Top
    if y_value < resolution - 1:
        self.rect = Rectangle(pos=(step * x_value, (step * y_value) + 450 + step),
                              size=(self.x_size / resolution, self.y_size / resolution))
        self.pixels_binary[x_value][y_value + 1] = 1.0

    # Top left
    if y_value < resolution - 1 and x_value > 0:
        if y_value < resolution - 1:
            self.rect = Rectangle(pos=(step * x_value - step, (step * y_value) + 450 + step),
                                  size=(self.x_size / resolution, self.y_size / resolution))
            self.pixels_binary[x_value - 1][y_value + 1] = 1.0


class CanvasApp(App):
    def build(self):
        return Touch()


if __name__ == '__main__':
    CanvasApp().run()
