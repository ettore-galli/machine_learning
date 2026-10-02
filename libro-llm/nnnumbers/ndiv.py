def plotdiv(n: int, d: int) -> list[str]:

    residual = n
    plot = []

    while residual > d:
        plot.append("*" * d)
        residual = residual - d

    plot.append("*" * residual)

    return plot


if __name__ == "__main__":
    for id, line in enumerate(plotdiv(221, 7)):
        print(str(id).rjust(3), line)
