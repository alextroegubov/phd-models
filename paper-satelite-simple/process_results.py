import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


B_MAX_RANGE = [8, 16, 100]


def process_validation(path: Path):
    ana_files = path.glob("ana-*.csv")
    sim_files = path.glob("sim-*.csv")

    ana_data = []
    sim_data = []

    for file in ana_files:
        data = pd.read_csv(file, header=None)
        ana_data.append(data.to_numpy()[0])

    for file in sim_files:
        data = pd.read_csv(file, header=None)
        sim_data.append(data.to_numpy()[0])

    ana_data = pd.DataFrame(
        ana_data, columns=["v", "rt_rej_prob_1", "rt_rej_prob_2", "data_rej_prob"]
    )
    ana_data["v"] = ana_data["v"].astype(int)
    ana_data.sort_values(by="v", inplace=True)
    sim_data = pd.DataFrame(
        sim_data, columns=["v", "rt_rej_prob_1", "rt_rej_prob_2", "data_rej_prob"]
    )
    sim_data["v"] = sim_data["v"].astype(int)

    ana_data = ana_data[ana_data.v <= 50]
    sim_data = sim_data[sim_data.v <= 50]

    plt.figure()  # figsize=(10, 8))
    # analytical result: has lines without marker, of different colors
    plt.plot(ana_data["v"], ana_data["rt_rej_prob_1"], label=r"$\pi_1$, ан.м.", linestyle="-")
    plt.plot(ana_data["v"], ana_data["rt_rej_prob_2"], label=r"$\pi_2$, ан.м.", linestyle="-")
    plt.plot(ana_data["v"], ana_data["data_rej_prob"], label=r"$\pi_e$, ан.м.", linestyle="-")

    # simulation result: has only markers of different colors
    plt.plot(
        sim_data["v"],
        sim_data["rt_rej_prob_1"],
        label=r"$\pi_1$, им.м",
        linestyle="",
        marker="o",
    )
    plt.plot(
        sim_data["v"],
        sim_data["rt_rej_prob_2"],
        label=r"$\pi_2$, им.м",
        linestyle="",
        marker="*",
    )
    plt.plot(
        sim_data["v"],
        sim_data["data_rej_prob"],
        label=r"$\pi_e$, им.м",
        linestyle="",
        marker="d",
    )

    plt.xlabel(r"Число единиц канального ресурса, $v$")
    plt.yscale("log")
    plt.ylabel("Вероятность отказа в обслуживании")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(path.parent / "validation.png")


def process_qos(path: Path):
    ana_files = path.glob("ana-*.csv")
    ana_data = []
    for file in ana_files:
        data = pd.read_csv(file, header=None)
        ana_data.append(data.to_numpy()[0])

    ana_data = pd.DataFrame(ana_data, columns=["v", "b_max", "pi_1", "pi_2", "pi_e", "W", "b_e"])
    ana_data["v"] = ana_data["v"].astype(int)
    ana_data["b_max"] = ana_data["b_max"].astype(int)
    ana_data["max_pi"] = ana_data[["pi_1", "pi_2", "pi_e"]].max(axis=1)
    ana_data.sort_values(by="v", inplace=True)

    ana_data = ana_data[(ana_data.v <= 80)]

    def plot_value(ana_data, name, label_an):
        # markers = ["*", "o", "v"]
        for i, b_max in enumerate([8, 16]):
            plt.plot(
                ana_data["v"][ana_data.b_max == b_max],
                ana_data[name][ana_data.b_max == b_max],
                label=label_an + r", $b_{max}$ = " + str(b_max),
                linestyle="-",
            )
        plt.plot(
            ana_data["v"][ana_data.b_max == 100],
            ana_data[name][ana_data.b_max == 100],
            label=label_an + r", $b_{max}$ = " + str(100),
            linestyle="-",
        )

    xlabel = r"Канальный ресурс $v$"
    # pi_e approximation
    plt.figure()

    plot_value(ana_data[ana_data.v <= 60], "max_pi", label_an=r"$\pi_{max}$")

    plt.xlabel(xlabel)
    plt.yscale("log")
    plt.ylabel("Максимальная доля потерянных заявок")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(path.parent / "qos_pi_max.png")

    # mean service time approximation
    plt.figure()

    plot_value(ana_data, "W", label_an=r"$W$")

    plt.xlabel(xlabel)
    plt.ylabel("Среднее время обслуживания заявки потока ЭТ")
    plt.yticks([0, 0.1, 0.125, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(path.parent / "qos_W.png")

    # mean resources per request
    plt.figure()
    plot_value(ana_data, "b_e", label_an=r"$b_e$")

    plt.xlabel(xlabel)
    plt.ylabel("Среднее число единиц ресурса на заявку потока ЭТ")
    plt.yticks([0, 8, 10, 16, 20, 30, 40])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(path.parent / "qos_b_e.png")


# def process_approximation_analysis(path: Path):
#     ana_files = path.glob("ana-*.csv")
#     approx_files = path.glob("approx-*.csv")

#     ana_data = []
#     approx_data = []

#     for file in ana_files:
#         data = pd.read_csv(file, header=None)
#         ana_data.append(data.to_numpy()[0])

#     for file in approx_files:
#         data = pd.read_csv(file, header=None)
#         approx_data.append(data.to_numpy()[0])

#     ana_data = pd.DataFrame(ana_data, columns=["v", "b_max", "pi_e_1", "y_2", "W_2", "b_e_2"])
#     ana_data["v"] = ana_data["v"].astype(int)
#     ana_data.sort_values(by="v", inplace=True)

#     approx_data = pd.DataFrame(approx_data, columns=["v", "b_max", "pi_e_1", "y_2", "W_2", "b_e_2"])
#     approx_data["v"] = approx_data["v"].astype(int)

#     ana_data = ana_data[(ana_data.v <= 60) & (ana_data.v >= 25)]
#     approx_data = approx_data[(approx_data.v <= 60) & (approx_data.v >= 25)]

#     def plot_value(ana_data, approx_data, name, label_an, label_approx):
#         markers = ["*", "o", "v"]
#         for i, b_max in enumerate([4, 8]):
#             plt.plot(
#                 ana_data["v"][ana_data.b_max == b_max],
#                 ana_data[name][ana_data.b_max == b_max],
#                 label=label_an + r", $b_{max}$ = " + str(b_max) + ", ан.м.",
#                 linestyle="-",
#             )
#             plt.plot(
#                 approx_data["v"][approx_data.b_max == b_max],
#                 approx_data[name][approx_data.b_max == b_max],
#                 label=label_approx + r", $b_{max}$ = " + str(b_max) + ", прибл.",
#                 linestyle="",
#                 marker=markers[i],
#             )
#         plt.plot(
#             ana_data["v"][ana_data.b_max == ana_data.v],
#             ana_data[name][ana_data.b_max == ana_data.v],
#             label=label_an + r", $b_{max}$ = " + str(100) + ", ан.м.",
#             linestyle="-",
#         )
#         plt.plot(
#             approx_data["v"][approx_data.b_max == approx_data.v],
#             approx_data[name][approx_data.b_max == approx_data.v],
#             label=label_approx + r", $b_{max}$ = " + str(100) + ", прибл.",
#             linestyle="",
#             marker=markers[-1],
#         )

#     # xlabel = r"Потенциальная загрузка единицы ресурса, $\rho$"
#     xlabel = r"Канальный ресурс $v$"
#     # pi_e approximation
#     plt.figure()

#     plot_value(ana_data, approx_data, "pi_e_1", label_an=r"$\pi_e$", label_approx=r"$\pi_e^*$")

#     plt.xlabel(xlabel)
#     plt.yscale("log")
#     plt.ylabel("Вероятность отказа в обслуживании")
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(path.parent / "approximation_pi_e.png")

#     # mean service time approximation
#     plt.figure()

#     plot_value(ana_data, approx_data, "W_2", label_an=r"$W$", label_approx=r"$W^*$")

#     plt.xlabel(xlabel)
#     plt.ylabel("Среднее время обслуживания заявки потока ЭТ")
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(path.parent / "approximation_W.png")

#     # mean requests in service approximation
#     plt.figure()
#     plot_value(ana_data, approx_data, "y_2", label_an=r"$y_e$", label_approx=r"$y_e^*$")

#     plt.xlabel(xlabel)
#     plt.ylabel("Среднее число заявок потока ЭТ в системе")
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(path.parent / "approximation_y_e.png")

#     # mean resources per request
#     plt.figure()
#     plot_value(ana_data, approx_data, "b_e_2", label_an=r"$b_e$", label_approx=r"$b_e^*$")

#     plt.xlabel(xlabel)
#     plt.ylabel("Среднее число единиц ресурса на заявку потока ЭТ")
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(path.parent / "approximation_b_e.png")


if __name__ == "__main__":
    process_validation(
        Path("/home/alex/wsl/phd-queue/phd-models/paper-satelite-simple/results/validation")
    )
    process_qos(
        Path("/home/alex/wsl/phd-queue/phd-models/paper-satelite-simple/results/qos")
    )
