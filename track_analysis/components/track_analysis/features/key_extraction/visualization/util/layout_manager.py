class LayoutManager:
    @staticmethod
    def grid(n_plots: int, max_cols: int = 4) -> (int, int):
        cols = min(n_plots, max_cols)
        rows = (n_plots + cols - 1) // cols
        return rows, cols
