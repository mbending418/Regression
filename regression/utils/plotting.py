import math


def auto_size_subplots(subplot_count: int) -> tuple[int, int]:
    """
    automatically determine the optimal subplot size from how many subplots there are

    :param subplot_count: how many subplots you need
    :return: the subplot size
    """

    max_diff = math.ceil(math.log10(subplot_count))
    for base_plot_size in range(0, math.ceil(math.sqrt(subplot_count))):
        for diff in range(max_diff):
            if base_plot_size * (base_plot_size + max_diff) >= subplot_count:
                return base_plot_size + max_diff, base_plot_size

    return math.ceil(math.sqrt(subplot_count)), math.ceil(math.sqrt(subplot_count))
