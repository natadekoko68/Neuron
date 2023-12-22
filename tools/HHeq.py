import matplotlib.pyplot as plt
import numpy as np

# import japanize_matplotlib

E_REST = -65.0
C = 1.0
G_LEAK = 0.3
E_LEAK = 10.6 + E_REST
G_NA = 120.0
E_NA = 115.0 + E_REST
G_K = 36.0
E_K = -12.0 + E_REST

DT = 0.01
T = 100
NT = int(T / DT)


def alpha_m(v):
    return (2.5 - 0.1 * (v - E_REST)) / (np.exp(2.5 - 0.1 * (v - E_REST)) - 1)


def beta_m(v):
    return 4 * np.exp(-(v - E_REST) / 18)


def alpha_h(v):
    return 0.07 * np.exp(-(v - E_REST) / 20)


def beta_h(v):
    return 1 / (np.exp(3 - 0.1 * (v - E_REST)) + 1)


def alpha_n(v):
    return (0.1 - 0.01 * (v - E_REST)) / (np.exp(1 - 0.1 * (v - E_REST)) - 1)


def beta_n(v):
    return 0.125 * np.exp(- (v - E_REST) / 80)


def m0(v):
    return alpha_m(v) / (alpha_m(v) + beta_m(v))


def h0(v):
    return alpha_h(v) / (alpha_h(v) + beta_h(v))


def n0(v):
    return alpha_n(v) / (alpha_n(v) + beta_n(v))


def tau_m(v):
    return 1.0 / (alpha_m(v) + beta_m(v))


def tau_h(v):
    return 1.0 / (alpha_h(v) + beta_h(v))


def tau_n(v):
    return 1.0 / (alpha_n(v) + beta_n(v))


def dmdt(v, m):
    return (1.0 / tau_m(v)) * (-m + m0(v))


def dhdt(v, h):
    return (1.0 / tau_h(v)) * (-h + h0(v))


def dndt(v, n):
    return (1.0 / tau_n(v)) * (-n + n0(v))


def dvdt(v, m, h, n, i_ext):
    return (-G_LEAK * (v - E_LEAK) - G_NA * (m ** 3) * h * (v - E_NA) - G_K * (n ** 4) * (v - E_K) + i_ext) / C


def main():
    rec = np.zeros((NT, 2))

    v = E_REST
    m = m0(v)
    h = h0(v)
    n = n0(v)

    i_ext = 9.0

    for nt in range(NT):
        t = DT * nt

        rec[nt, 0] = t
        rec[nt, 1] = v

        print(f"{t:.4f}, {v:.4f}, {m:.4f}, {h:.4f}, {n:.4f}")

        dmdt1 = dmdt(v, m)
        dhdt1 = dhdt(v, h)
        dndt1 = dndt(v, n)
        dvdt1 = dvdt(v, m, h, n, i_ext)

        dmdt2 = dmdt(v + 0.5 * DT * dvdt1, m + 0.5 * DT * dmdt1)
        dhdt2 = dhdt(v + 0.5 * DT * dvdt1, h + 0.5 * DT * dhdt1)
        dndt2 = dndt(v + 0.5 * DT * dvdt1, n + 0.5 * DT * dndt1)
        dvdt2 = dvdt(v + 0.5 * DT * dvdt1, m + 0.5 * DT * dmdt1, h + 0.5 * DT * dhdt1, n + 0.5 * DT * dndt1, i_ext)

        dmdt3 = dmdt(v + 0.5 * DT * dvdt2, m + 0.5 * DT * dmdt2)
        dhdt3 = dhdt(v + 0.5 * DT * dvdt2, h + 0.5 * DT * dhdt2)
        dndt3 = dndt(v + 0.5 * DT * dvdt2, n + 0.5 * DT * dndt2)
        dvdt3 = dvdt(v + 0.5 * DT * dvdt2, m + 0.5 * DT * dmdt2, h + 0.5 * DT * dhdt2, n + 0.5 * DT * dndt2, i_ext)

        dmdt4 = dmdt(v + DT * dvdt3, m + DT * dmdt3)
        dhdt4 = dhdt(v + DT * dvdt3, h + DT * dhdt3)
        dndt4 = dndt(v + DT * dvdt3, n + DT * dndt3)
        dvdt4 = dvdt(v + DT * dvdt3, m + DT * dmdt3, h + DT * dhdt3, n + DT * dndt3, i_ext)

        m += DT * (dmdt1 + 2 * dmdt2 + 2 * dmdt3 + dmdt4) / 6
        h += DT * (dhdt1 + 2 * dhdt2 + 2 * dhdt3 + dhdt4) / 6
        n += DT * (dndt1 + 2 * dndt2 + 2 * dndt3 + dndt4) / 6
        v += DT * (dvdt1 + 2 * dvdt2 + 2 * dvdt3 + dvdt4) / 6

    plt.plot(rec[:, 0], rec[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
