from manim import *
import numpy as np

from math import sin, cos

class Legend(Scene):
    def construct(self):
        t = Tex("t = \# tickets in bucket").shift(UP)
        v = Tex("v = value of prize")
        s = Tex("s = dollars / ticket")
        line = Line().scale(4)
        self.play(Write(line))
        self.play(Write(t.next_to(line, DOWN)))
        self.play(Write(v.next_to(t, DOWN)))
        self.play(Write(s.next_to(v, DOWN)))
        line1 = Line().scale(4)
        self.play(Write(line1.next_to(s, DOWN)))
        t_val = MathTex("=20")
        v_val = MathTex("=100")
        s_val = MathTex("=2")
        self.play(Write(t_val.next_to(t, RIGHT)))
        self.play(Write(v_val.next_to(v, RIGHT)))
        self.play(Write(s_val.next_to(s, RIGHT)))
        self.wait(1)



            
class ProbabilityAndPayout(Scene):
    def construct(self):
        profit = Tex("Profit")
        probability = Tex("Probability")
        payout = Tex("Payout")
        self.play(Write(profit.shift(UP)))
        self.play(Write(probability.next_to(profit, DOWN * 3 + LEFT)), Write(payout.next_to(profit, DOWN * 3 + RIGHT)))
        
        probability_formula = MathTex(r"\left(\frac{x}{x + t}\right)")
        self.play(Write(probability_formula.next_to(probability, DOWN * 2).shift(RIGHT)))
        payout_formula = MathTex("(v-xs)")
        self.play(Write(payout_formula.next_to(probability_formula, RIGHT * 3)))

        self.play(Write(Tex("profit(x) = ").next_to(probability_formula, LEFT)), Write(MathTex("*").next_to(probability_formula, RIGHT)))

class GraphingProfit(Scene):
    def construct(self):
        equation = MathTex(r"f(x)=\left(\frac{x}{x + t}\right)*(v-xs)")
        self.play(Write(equation.shift(UP * 3 + RIGHT * 3)))

        ax = Axes(x_range=[0, 52, 10], y_range=[0, 50, 10], axis_config = {
            "numbers_to_include": np.arange(0, 50, 10)
        })
        labels = ax.get_axis_labels(x_label="x", y_label="profit")

        t = 20
        v = 100
        s = 2
        graph = ax.plot(lambda x: (x / (x + t)) * (v - x * s), color = RED)

        self.play(Create(ax), Write(labels))
        self.play(Write(graph))
        self.wait(2)
        self.play(Create(Dot(color=BLUE).move_to(ax.c2p(17.41657, 30.3337))))

        alpha_tracker = ValueTracker(0.5)

        tangent = TangentLine(graph, alpha = alpha_tracker.get_value(), length = 4)
        tangent.add_updater(
            lambda x: x.become(TangentLine(graph, alpha = alpha_tracker.get_value(), length = 4))
        )

        self.play(Create(tangent))
        self.play(alpha_tracker.animate.set_value(.7))
        self.wait(.5)
        self.play(alpha_tracker.animate.set_value(.2))
        self.wait(.5)
        self.play(alpha_tracker.animate.set_value(0.39))
        self.wait(3)

class OptimizingProfit(Scene):
    def construct(self):
        equations = [MathTex(r"f(x)=\left(\frac{x}{x + t}\right)*(v-xs)"),
                     MathTex(r"f(x)=\frac{vx-sx^2}{x+t}"),
                     MathTex(r"\frac{d}{dx}uv^{-1}=\frac{u'v-uv'}{v^2}"),
                     MathTex(r"f'(x)=\frac{(v-2sx)(x+t)-(vx-sx^2)}{(x+t)^2}"),
                     MathTex(r"f'(x)=\frac{vt-2sxt-sx^2}{(x+t)^2}")]
        self.play(Write(equations[0].shift(UP * 2.5)))
        for i, equation in enumerate(equations[1:]):
            self.play(Write(equation.next_to(equations[i], DOWN)))

class QuadraticFormula(Scene):
    def construct(self):
        equations = [MathTex(r"f'(x)=\frac{vt-2sxt-sx^2}{(x+t)^2}=0"),
                     MathTex(r"vt-2sxt-sx^2=0"),
                     Text("Of the form ax^2 + bx + c = 0!").scale(.66),
                     MathTex(r"x=\frac{-b+-\sqrt{b^2-4ac}}{2a}"),
                     MathTex(r"x=\frac{2st-\sqrt{4s^2t^2 + 4tvs}}{-2s}")]
        
        
        self.play(Write(equations[0].shift(UP * 2.5)))
        for i, equation in enumerate(equations[1:]):
            self.play(Write(equation.next_to(equations[i], DOWN)))
        self.wait(4)
        
class Vectorization(Scene):
    def construct(self):
        t = MathTex(r"t")
        v = MathTex(r"v")
        s = MathTex(r"s")

        self.play(Write(t.move_to(UP * 3 + LEFT * 2)))
        self.play(Write(v.move_to(UP * 3)))
        self.play(Write(s.move_to(UP * 3 + RIGHT * 2)))

        self.play(t.animate.become(MathTex(r"\overrightarrow{t}").move_to(UP * 3 + LEFT * 2)))
        self.play(v.animate.become(MathTex(r"\overrightarrow{v}").move_to(UP * 3)))

        t_vector = MathTex(r"\begin{bmatrix}51\\22\\17\\19\\13\\30\\24\\19\\24\\12\end{bmatrix}")
        v_vector = MathTex(r"\begin{bmatrix}\$100\\\$50\\\$60\\\$30\\\$30\\\$70\\\$40\\\$60\\\$50\\\$30\end{bmatrix}")
        self.play(Write(t_vector.next_to(t, DOWN)))
        self.play(Write(v_vector.next_to(v, DOWN)))
        self.play(Write(MathTex(r"\$2").next_to(s, DOWN * 2)))
        self.wait(1)

class Formalization(Scene):
    def construct(self):
        equations = [MathTex(r"f(\overrightarrow{x})=\sum_{i=1}^{n}\left(\frac{x_i}{x_i + t_i}\right)*(v_i-x_is)"),
                     MathTex(r"g(\overrightarrow{x})=\sum_{i=1}^{n}x_i=c"),
                     Text("Where c = 50 (we have 50 tickets total)").scale(.5)]
        
        self.play(Write(equations[0].shift(UP * 2.5)))
        for i, equation in enumerate(equations[1:]):
            self.play(Write(equation.next_to(equations[i], DOWN)))
        self.play(Write(Text("Maximize f, given that g = c").next_to(equations[-1], DOWN * 4)))
        self.wait(4)

class Gradients(Scene):
    def construct(self):
        equations = [
            MathTex(r"\nabla f(\overrightarrow{x})=[\frac{\partial f}{\partial x_i} \text{ for i in 1...n}]"),
            MathTex(r"\frac{\partial f}{\partial x_i} = \frac{v_i t_i-sx_i^2-2sx_i t_i}{(x_i+t_i)^2}"),
            MathTex(r"g(\overrightarrow{x})=x_1 + x_2 + \hdots + x_n"),
            MathTex(r"\nabla g(\overrightarrow{x})=\begin{bmatrix}1, 1, \hdots, 1\end{bmatrix}")
        ]

        self.play(Write(equations[0].shift(UP * 1.5)))
        for i, equation in enumerate(equations[1:]):
            self.play(Write(equation.next_to(equations[i], DOWN)))
        self.wait(4)

class SystemOfEquations(Scene):
    def construct(self):
        general = MathTex(r"\nabla f = \lambda * \nabla g,\text{   } g = c = 50")
        system = MathTex(r"\begin{cases} \frac{\partial f}{\partial x_1} = 1 * \lambda \\ \frac{\partial f}{\partial x_2} = 1 * \lambda \\ \vdots \\ \frac{\partial f}{\partial x_{10}} = 1 * \lambda \\ x_1 + x_2 + \hdots + x_{10} = 50\end{cases}")
        self.play(Write(general.move_to(UP * 2)))
        self.play(Write(system.next_to(general, DOWN)))
        self.wait(1)

# https://slama.dev/manim/plotting-and-3d-scenes/
class GradientDescent(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        x_label = axes.get_x_axis_label(MathTex("x"))
        y_label = axes.get_y_axis_label(MathTex("y")).shift(UP * 2)

        self.set_camera_orientation(zoom = 0.5)
        self.renderer.camera.light_source.move_to(5*UP)
        self.play(Create(axes), Create(x_label), Create(y_label))
        self.wait(1)

        self.move_camera(phi = 60 * DEGREES, theta = 30 * DEGREES, zoom = 1, run_time = 3)
        self.wait(1)

        def z(x, y):
            return (x**2 + y**2) / 20
        
        def grad(x, y):
            return (2/7 * x, 2/7 * y)
        
        surface = Surface(
            lambda u, v: axes.c2p(u, v, z(u, v)),
            resolution = [50, 50],
            u_range = [-5, 5],
            v_range = [-5, 5],
            fill_color = BLUE,
            stroke_color = BLUE,
            shade_in_3d = True,
            fill_opacity = 1,
            checkerboard_colors = [],
        )

        self.play(Write(surface))
        self.wait(5)

        coords = [-4, 3]
        dot = Dot().move_to(axes.c2p(*coords, z(*coords)))
        self.play(Create(dot))
        for _ in range(5):
            dx, dy = grad(*coords)
            arrow = Arrow(start = axes.c2p(*coords, z(*coords)), end = axes.c2p(coords[0] + dx, coords[1] + dy, z(coords[0] + dx, coords[1] + dy)))
            self.play(Create(arrow))
            self.wait(.5)
            self.play(arrow.animate.become(Arrow(start = axes.c2p(*coords, z(*coords)), end = axes.c2p(coords[0] - dx, coords[1] - dy, z(coords[0] - dx, coords[1] - dy)))))
            self.wait(.1)
            self.play(Uncreate(arrow))
            coords[0] -= dx
            coords[1] -= dy
            self.play(dot.animate.move_to(axes.c2p(*coords, z(*coords))))

class PreWarp(Scene):
    def construct(self):
        axes = Axes(x_range=[0, 6], y_range=[0, 6], x_length = 5, y_length = 5)
        x_label = axes.get_x_axis_label(Tex("x"))
        y_label = axes.get_y_axis_label(Tex("y"))
        self.play(Write(axes))
        self.wait(1)
        
        coords = [3, 2]

        eq = MathTex(r"x+y=5")
        self.play(Write(eq.next_to(axes, UP)))
        distance = MathTex(r"\text{Distance from constraint: } x+y-5 = 0.00").next_to(axes, DOWN)
        distance_tracker = ValueTracker(0)
        distance.add_updater(
            lambda x: x.become(MathTex(r"\text{Distance from constraint: } x+y-5 = " + f"{distance_tracker.get_value():.2f}").next_to(axes, DOWN))
        )

        self.play(Write(distance))

        self.wait(1)
        graph = axes.plot(
            lambda x: -x + 5,
            color = RED,
            x_range = [0, 5]
        )
        self.play(Write(graph))
        
        self.wait(1)

        dot = Dot(color = BLUE).move_to(axes.c2p(*coords))
        self.play(Create(dot))
        self.wait(1)

        other_coords = [
            [4, 5],
            [1, 2],
            [1, 4],
            [5, 4],
            [3, 2]
        ]

        for other_coord in other_coords:
            self.play(distance_tracker.animate.set_value(abs(other_coord[0] + other_coord[1] - 5)), dot.animate.move_to(axes.c2p(*other_coord)))
            self.wait(0.5)

class Warp(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        x_label = axes.get_x_axis_label(MathTex("x"))
        y_label = axes.get_y_axis_label(MathTex("y")).shift(UP * 2)
        self.set_camera_orientation(phi = 50 * DEGREES, theta = -25 * DEGREES)
        eq0 = MathTex(r"\text{Distance measure (unitless): }x+y-5")
        eq1 = MathTex(r"\text{Constraint trough (not to scale): }(x+y-5)^2").scale(0.66)

        self.set_camera_orientation(zoom = 0.55)
        self.play(Create(axes), Create(x_label), Create(y_label))

        distance_curve = ParametricFunction(
            lambda t: axes.c2p(
                    5-5*t, 5*t, 0
                ),
            color = RED,
            t_range = (1, 0, -0.01)
        ).set_shade_in_3d(True)

        constraint_trough = Surface(
            lambda u, v: axes.c2p(
                u, v, (u + v - 5) ** 2 / 5
            ),
            resolution = [50, 50],
            u_range = [0, 5],
            v_range = [0, 5],
            fill_color = RED,
            stroke_color = RED,
            shade_in_3d = True,
            fill_opacity = 1,
            checkerboard_colors = [],
        )

        self.begin_ambient_camera_rotation(rate = 0.1)

        self.add_fixed_in_frame_mobjects(eq0.to_corner(UL))
        self.play(Write(distance_curve))
        self.wait(3)
        self.add_fixed_in_frame_mobjects(eq1.next_to(eq0, DOWN))
        self.play(Unwrite(distance_curve))
        self.play(Create(constraint_trough))
        # self.add(constraint_trough)
        self.wait(6)
        self.play(Unwrite(constraint_trough))
        # self.remove(constraint_trough)
        
        self.stop_ambient_camera_rotation()

        self.play(Unwrite(eq0), Unwrite(eq1))
        eq2 = MathTex(r"\text{Distance measure (unitless): }x^2 + y^2 - 4")
        eq3 = MathTex(r"\text{Constraint trough (not to scale): }(x^2+y^2-4)^2").scale(0.66)

        distance_curve1 = ParametricFunction(
            lambda t: axes.c2p(
                    2 * sin(t), 2 * cos(t), 0
                ),
            color = BLUE,
            t_range = (0, 2*3.15, 0.01)
        ).set_shade_in_3d(True)

        z = lambda u,v: ((u**2 + v**2 - 4) ** 2) / 5
        constraint_trough1 = Surface(
            lambda u, v: axes.c2p(
                u, v, z(u, v) if z(u,v) <= 3.5 else 3.5
            ),
            resolution = [100, 100],
            u_range = [-3, 3],
            v_range = [-3, 3],
            fill_color = BLUE,
            stroke_color = BLUE,
            shade_in_3d = True,
            fill_opacity = 1,
            checkerboard_colors = [],
        )

        self.add_fixed_in_frame_mobjects(eq2.to_corner(UL))
        self.begin_ambient_camera_rotation(rate = 0.1)
        self.wait(1)
        self.play(Write(distance_curve1))
        self.wait(3)
        self.add_fixed_in_frame_mobjects(eq3.next_to(eq2, DOWN))
        self.play(Unwrite(distance_curve1))
        self.play(Create(constraint_trough1))
        # self.add(constraint_trough1)
        self.wait(6)
        self.play(Unwrite(constraint_trough1))
        # self.add(constraint_trough1)
        
class Overlay(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        x_label = axes.get_x_axis_label(MathTex("x"))
        y_label = axes.get_y_axis_label(MathTex("y")).shift(UP * 2)

        self.set_camera_orientation(phi = 60 * DEGREES, theta = 30 * DEGREES, zoom = 0.5)
        self.renderer.camera.light_source.move_to(5*UP)
        self.play(Create(axes), Create(x_label), Create(y_label))
        self.wait(1)
        self.move_camera(theta = -30*DEGREES, run_time = 2)

        self.wait(1)

        old_eq = MathTex(r"\text{Old equation (not to scale): } x^2 + y^2")
        new_eq = MathTex(r"\text{New equation: } (x^2 + y^2) + (x+y-5)^2")

        old_surface = Surface(
            lambda u, v: axes.c2p(
                u, v, (u**2 + v**2) / 20
            ),
            resolution = [50, 50],
            u_range = [-5, 5],
            v_range = [-5, 5],
            fill_color = BLUE,
            stroke_color = BLUE,
            shade_in_3d = True,
            fill_opacity = 1,
            checkerboard_colors = [],
        )
        distance_curve = ParametricFunction(
            lambda t: axes.c2p(
                    5-5*t, 5*t, ((5-5*t)**2 + (5*t)**2) / 20 + .25
                ),
            color = RED,
            t_range = (1, 0, -0.01)
        ).set_shade_in_3d(True)
        z = lambda u, v: (u**2 + v**2) / 20 + (u + v - 5) ** 2 / 15
        new_surface = Surface(
            lambda u, v: axes.c2p(
                u, v, z(u, v) if z(u, v) < 80 else 80
            ),resolution = [50, 50],
            u_range = [-5, 5],
            v_range = [-5, 5],
            fill_color = PURPLE,
            stroke_color = PURPLE,
            shade_in_3d = True,
            fill_opacity = 1,
            checkerboard_colors = [],
        )

        self.add_fixed_in_frame_mobjects(old_eq.to_corner(UL))
        self.wait(1)
        self.play(Write(old_surface))
        self.wait(1)
        self.wait(1)
        self.play(Write(distance_curve))
        self.play(Unwrite(distance_curve))
        self.remove(old_eq)
        self.add_fixed_in_frame_mobjects(new_eq.to_corner(UL))
        self.play(old_surface.animate.become(new_surface))
        self.add(new_surface)
        self.remove(old_surface)
        self.wait(1)

        self.begin_ambient_camera_rotation(rate = 0.1)
        self.wait(7)

class Reward(Scene):
    def construct(self):
        equations = [MathTex(r"f(\overrightarrow{x})=\sum_{i=1}^{n}\left(\frac{x_i}{x_i + t_i}\right)*(v_i-x_is)"),
                    MathTex(r"g(\overrightarrow{x})=\sum_{i=1}^{n}x_i=c"),
                    MathTex(r"\text{reward}=r(\overrightarrow{x})=f(\overrightarrow{x})-g(\overrightarrow{x})^2"),
                    MathTex(r"\nabla r(\overrightarrow{x})=[\frac{\partial f}{\partial x_i}-2g(\overrightarrow{x})\frac{\partial g}{\partial x_i} \text{ for i in 1...n}]"),
                    MathTex(r"\nabla r(\overrightarrow{x})=[\frac{\partial f}{\partial x_i}-2g(\overrightarrow{x}) \text{ for i in 1...n}]")
        ]

        self.play(Write(equations[0].move_to(UP * 2.5)))
        for i, equation in enumerate(equations[1:]):
            self.play(Write(equation.next_to(equations[i], DOWN)))
        self.wait(2)

class GoBackToGradientDescent(Scene):
    def construct(self):
        axes = Axes(x_range = [0, 5], y_range = [0, 5])
        self.play(Create(axes))

        f = lambda x: (x-2.5)**2
        f_prime = lambda x: 2*(x-2.5)
        get_alpha = lambda x: (x - .5) / 4
        graph = axes.plot(f, color = RED, x_range = [0.5, 4.5])
        self.play(Write(graph))

        x = 1
        x_tracker = ValueTracker(x)
        
        dot = Dot(color = BLUE).move_to(axes.c2p(x, f(x)))
        dot.add_updater(
            lambda mob: mob.become(Dot(color = BLUE).move_to(axes.c2p(x_tracker.get_value(), f(x_tracker.get_value()))))
        )

        # https://www.reddit.com/r/manim/comments/tfckvz/tangent_line_of_a_point/
        tangent = always_redraw(
            lambda: axes.get_secant_slope_group(
                x=x_tracker.get_value(),
                graph=graph,
                dx=0.01,
                secant_line_color=WHITE,
                secant_line_length=4,
            )
        )
        self.play(Create(dot), Create(tangent))

        eq = MathTex(r"\text{Slope: } " + str(round(f_prime(x_tracker.get_value()), 2))).to_corner(UR)
        eq.add_updater(
            lambda mob: mob.become(MathTex(r"\text{Slope: } " + str(round(f_prime(x_tracker.get_value()), 2))).to_corner(UR))
        )

        def get_slope():
            return round(f_prime(x_tracker.get_value()), 2)
        direction = MathTex(r"\text{Move } " + str(abs(get_slope())) + r"* \text{scalar to the " + ("left" if get_slope() >= 0 else "right") + "}").scale(0.5).next_to(eq, DOWN)
        direction.add_updater(
            lambda mob: mob.become(MathTex(r"\text{Move } " + str(abs(get_slope())) + r"* \text{scalar to the " + ("left" if get_slope() >= 0 else "right") + "}").scale(0.5).next_to(eq, DOWN))
        )

        self.play(Write(eq))
        self.play(Write(direction))
        self.wait(2)

        self.play(x_tracker.animate.set_value(4))
        self.wait(2)
        self.play(x_tracker.animate.set_value(2))
        self.wait(2)
        self.play(x_tracker.animate.set_value(3))
        self.wait(1)



