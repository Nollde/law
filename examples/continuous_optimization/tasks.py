# coding: utf-8

from pathlib import Path
import os
from time import sleep

from skopt.plots import plot_objective, plot_evaluations, plot_convergence
import matplotlib.pyplot as plt
import luigi
from luigi.util import inherits
from luigi import IntParameter, FloatParameter, ChoiceParameter

import law

law.contrib.load("matplotlib")


class Task(law.Task):
    """
    Base that provides some convenience methods to create local file and
    directory targets at the default data path.
    """

    def local_path(self, *path):
        # ANALYSIS_DATA_PATH is defined in setup.sh
        parts = (os.getenv("ANALYSIS_DATA_PATH"), self.__class__.__name__) + path
        return os.path.join(*parts)

    def local_target(self, *path, **kwargs):
        return law.LocalFileTarget(self.local_path(*path), **kwargs)


class TargetLock(object):
    def __init__(self, target):
        self.target = target
        self.path = target.path
        self.lock = self.path + ".lock"

    def __enter__(self):
        while True:
            if Path(self.lock).is_file():
                sleep(1)
            else:
                Path(self.lock).touch()
                self.loaded = self.target.load()
                break
        return self.loaded

    def __exit__(self, type, value, traceback):
        self.target.dump(self.loaded)
        os.remove(self.lock)


class Optimizer(Task, law.LocalWorkflow):
    """
    Workflow that runs optimization.
    """

    iterations = luigi.IntParameter(default=4, description="Number of iterations")
    n_parallel = luigi.IntParameter(
        default=2, description="Number of parallel evaluations"
    )

    def create_branch_map(self):
        return list(range(self.n_parallel))

    def requires(self):
        return OptimizerPreparation.req(self)

    def output(self):
        return self.local_target("optimizer.pkl")

    @property
    def todo(self):
        return self.local_target("todo_{}.json".format(self.branch))

    def run(self):
        print(f"---------------- hello from branch: {self.branch} ----------------")
        if self.todo.exists():
            x = self.todo.load()
            outp = yield Objective.req(self, x=x)
            y = outp.load()["y"]
            with TargetLock(self.input()["opt"]) as opt:
                print("writing")
                print(f"x: {x}, y: {y}")
                opt.tell(x, y)
                self.todo.remove()

        with TargetLock(self.input()["opt"]) as opt, TargetLock(
            self.input()["initial_todos"]
        ) as initial_todos:
            if len(opt.Xi) >= self.iterations:
                self.output().dump(opt)
                return
            print("got new todo", end=", ")
            if len(initial_todos) > 0:
                x = initial_todos.pop(0)
                print("from initial todos", end=": ")
            else:
                x = opt.ask(n_points=1)[0]
                print("by asking", end=": ")
            print(x)
        self.todo.dump(x)
        output = yield Objective.req(self, x=x)


class OptimizerPreparation(Task):
    """
    Workflow that runs optimization.
    """

    n_initial_points = luigi.IntParameter(
        default=10,
        description="Number of random sampled values \
        before starting optimizations",
    )

    def output(self):
        return {
            "opt": self.local_target("optimizer.pkl"),
            "initial_todos": self.local_target("initial_todos.json"),
        }

    def run(self):
        import skopt

        optimizer = skopt.Optimizer(
            dimensions=[skopt.space.Real(-5.0, 10.0), skopt.space.Real(0.0, 15.0)],
            random_state=1,
            n_initial_points=self.n_initial_points,
        )
        x = optimizer.ask(n_points=self.n_initial_points)

        with self.output()["opt"].localize("w") as tmp:
            tmp.dump(optimizer)
        with self.output()["initial_todos"].localize("w") as tmp:
            tmp.dump(x)


@inherits(Optimizer)
class OptimizerPlot(Task):
    """
    Workflow that runs optimization and plots results.
    """

    def create_branch_map(self):
        return list(range(self.n_parallel))

    plot_objective = luigi.BoolParameter(
        default=True,
        description="Plot objective. \
        Can be expensive to evaluate for high dimensional input",
    )

    def requires(self):
        return Optimizer.req(self, branch=-1)

    def output(self):
        collection = {
            "evaluations": self.local_target("evaluations.pdf"),
            "convergence": self.local_target("convergence.pdf"),
        }

        if self.plot_objective:
            collection["objective"] = self.local_target("objective.pdf")

        return collection

    def run(self):
        from skopt.plots import plot_objective, plot_evaluations, plot_convergence
        import matplotlib.pyplot as plt

        result = self.input()["collection"].targets[0].load().run(None, 0)
        output = self.output()

        plot_convergence(result)
        output["convergence"].dump(plt.gcf(), bbox_inches="tight")
        plt.close()
        plot_evaluations(result, bins=10)
        output["evaluations"].dump(plt.gcf(), bbox_inches="tight")
        plt.close()
        if self.plot_objective:
            plot_objective(result)
            output["objective"].dump(plt.gcf(), bbox_inches="tight")
            plt.close()


class Objective(Task):
    """
    Objective to optimize.

    This workflow will evaluate the branin function for given values `x`.
    In a real world example this will likely be a expensive to compute function like a
    neural network training or other computational demanding task.
    The workflow can be easily extended as a remote workflow to submit evaluation jobs
    to a batch system in order to run calculations in parallel.
    """

    x = luigi.ListParameter()

    def output(self):
        return self.local_target("x_{}.json".format(self.x))

    def run(self):
        from skopt.benchmarks import branin

        with self.output().localize("w") as tmp:
            tmp.dump({"x": self.x, "y": branin(self.x)})
