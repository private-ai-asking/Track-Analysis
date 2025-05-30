from matplotlib import pyplot as plt


# noinspection PyUnresolvedReferences,PyBroadException,PyUnboundLocalVariable
class WindowManager:
    @staticmethod
    def maximize(fig: plt.Figure) -> None:
        try:
            mgr = fig.canvas.manager
            mgr.window.state("zoomed")
        except Exception:
            try:
                mgr.window.showMaximized()
            except Exception:
                pass
