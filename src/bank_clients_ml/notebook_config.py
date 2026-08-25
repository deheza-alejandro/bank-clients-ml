from IPython import get_ipython


def setup_notebook():
    """
    Para detectar automáticamente cualquier cambio en los módulos sin necesidad de
    reiniciar el kernel
    """
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
