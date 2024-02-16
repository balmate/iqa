import matplotlib.pyplot as plt

def create_plot_for_metric(x: [], labelx: str, y: [], labely: str, title: str) -> None:
    # data
    plt.plot(x, y, "o-g")
    
    # axis names and title
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)

    # show
    plt.show()