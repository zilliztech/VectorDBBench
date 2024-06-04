from streamlit.components.v1 import iframe

from .. import extra


@extra
def jupyterlite(height: int = 600, width: int = 800):
    """
    Add a Jupyterlite sandbox to your Streamlit app

    Args:
        height (int): Height of the Jupyterlite sandbox. Default value is 600.
        width (int): Width of the Jupyterlite sandbox.  Default value is 800
    """

    iframe(
        src="https://jupyterlite.github.io/demo/repl/index.html?kernel=python&toolbar=1",
        height=height,
        width=width,
    )


def example():
    jupyterlite(1500, 1600)


__title__ = "Jupyterlite"
__desc__ = "Add a Jupyterlite sandbox to your Streamlit app in one command"
__icon__ = "💡"
__examples__ = [example]
__author__ = "Rahul Chauhan"
__experimental_playground__ = False
